"""Prepare native-chat recovery supervision and audit production readiness.

This module performs CPU data preparation only. It cannot launch training. Real
tokenizer rendering is required for prepared shards; the dry run is explicitly a
contract proof, never a claim of model, gradient, or modality capability.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "chimera_recovery.training_readiness.v1"
PACK_SCHEMA = "chimera_recovery.assistant_supervision.v1"
TEXT_VOCAB = 248_320
MUSIC_BEGIN = 252_416
MUSIC_CODES = 64_000
TOTAL_VOCAB = 317_952
HIDDEN = 5_120
FEATURES = 6_144
POSITIONS = 4
ROUTES = {
    **{r: ("D2", "assistant_text", ()) for r in (
        "text_instruction", "text_reasoning", "text_arithmetic", "text_code", "text_multiturn",
        "tool_schema", "tool_call", "tool_result", "tool_retry", "tool_parallel", "tool_multiturn")},
    "vision_grounding": ("D3d", "assistant_text+feature_kd", (6,)),
    "video_understand": ("D3d", "assistant_text+feature_kd", (6,)),
    "audio_understand": ("D3d", "assistant_text", (3,)),
    "image_generate": ("D3b", "guided_velocity", (1,)),
    "image_edit": ("D3b", "guided_velocity", (1,)),
    "audio_generate": ("D3b", "guided_velocity", (3,)),
    "music_generate": ("D3b", "guided_velocity", (4,)),
    "joint_av": ("D3b", "guided_velocity", (2, 3)),
    "volume_generate": ("D3b", "guided_velocity", (7,)),
}
ROUTE_ALIASES = {
    "text": "text_instruction", "instruction": "text_instruction", "code": "text_code",
    "reasoning": "text_reasoning", "tool": "tool_call", "vision_in": "vision_grounding",
    "image": "image_generate", "video_in": "video_understand", "music": "music_generate",
    "audio": "audio_generate", "h3": "joint_av", "volume": "volume_generate",
}
PINNED_FILES = (
    "PREREGISTRATION_OMNI_STAGE_D_v1.md", "tools/stage2_training_receipt.py",
    "tools/streamed_kd_trainer.py", "tools/d1_trunk_recovery.py", "tools/d3_hybrid_module.py",
    "queues/d1_launch.sh", "tools/queues/d3.sh",
)
STREAMED_BASELINE_SHA256 = "77091f8ab1581c1edac3e3f1c1e5d727d60f8fb8465745c03a6eb6b12f0567da"
D3_BASELINE_SHA256 = "540fca2d9d976da336c8d24ae5369c5a303ff37367cae54a9bcfc762a7e95197"


class ReadinessError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ReadinessError(f"{name}: expected lowercase sha256")
    return value


def route_name(value: str) -> str:
    resolved = ROUTE_ALIASES.get(value, value)
    if resolved not in ROUTES:
        raise ReadinessError(f"unsupported route {value!r}; no silent text fallback")
    return resolved


def source_group(record: dict[str, Any]) -> str | None:
    provenance = record.get("provenance", {})
    return provenance.get("recovery_group_id", provenance.get("group_id", record.get("group_id", record.get("group", record.get("id")))))


def check_json_schema(value: Any, schema: dict[str, Any], where: str) -> None:
    """Validate the conservative JSON-schema subset used by local tool fixtures.

    Unimplemented constraints refuse instead of appearing schema conditioned.
    """
    supported = {"type", "properties", "required", "additionalProperties", "items", "enum", "description", "title", "default"}
    if set(schema) - supported:
        raise ReadinessError(f"{where}: unsupported JSON-schema keywords {sorted(set(schema) - supported)}")
    kind = schema.get("type")
    kinds = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
             "string": lambda x: isinstance(x, str), "boolean": lambda x: isinstance(x, bool),
             "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
             "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x),
             "null": lambda x: x is None}
    if kind is not None and (kind not in kinds or not kinds[kind](value)):
        raise ReadinessError(f"{where}: value violates type {kind!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ReadinessError(f"{where}: value not in enum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ReadinessError(f"{where}: missing required arguments {sorted(missing)}")
        unknown = set(value) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ReadinessError(f"{where}: undeclared arguments {sorted(unknown)}")
        for key, item in value.items():
            if key in properties:
                check_json_schema(item, properties[key], f"{where}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            check_json_schema(item, schema["items"], f"{where}[{i}]")


def validate_messages(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ReadinessError("messages must be a nonempty native chat list")
    tools = record.get("tools", [])
    known = {}
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name")
        if tool.get("type") != "function" or not name or name in known:
            raise ReadinessError("tool declarations require unique function names")
        if function.get("parameters", {}).get("type") != "object":
            raise ReadinessError(f"tool {name}: parameters must declare an object")
        known[name] = function
    sup = record.get("supervision", {})
    if sup.get("assistant_only") is not True:
        raise ReadinessError("assistant_only supervision must be explicit")
    message_mask = sup.get("message_mask")
    if not isinstance(message_mask, list) or len(message_mask) != len(messages) or any(type(v) is not bool for v in message_mask):
        raise ReadinessError("message_mask must be booleans, one per native message")
    calls, pending, results = {}, set(), []
    selected = 0
    for i, (message, supervise) in enumerate(zip(messages, message_mask)):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ReadinessError(f"message {i}: unsupported role")
        if supervise and role != "assistant":
            raise ReadinessError(f"message {i}: prompt or tool result cannot be supervised")
        selected += int(supervise)
        if role == "assistant" and pending:
            raise ReadinessError("assistant continuation before every parallel tool result")
        if role == "user" and pending:
            raise ReadinessError("new user turn before outstanding tool results")
        for call in message.get("tool_calls", []):
            if role != "assistant" or call.get("type") != "function":
                raise ReadinessError("tool_calls belong to assistant function messages")
            call_id = call.get("id")
            function = call.get("function", {})
            name = function.get("name")
            if not isinstance(call_id, str) or not call_id or call_id in calls:
                raise ReadinessError("tool calls require unique nonempty ids")
            if name not in known:
                raise ReadinessError(f"call {call_id}: undeclared tool {name!r}")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ReadinessError(f"call {call_id}: malformed JSON arguments") from error
            if not isinstance(arguments, dict):
                raise ReadinessError(f"call {call_id}: arguments must be an object")
            check_json_schema(arguments, known[name]["parameters"], f"call {call_id}")
            calls[call_id] = name
            pending.add(call_id)
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ReadinessError("orphan, duplicate, or stale tool result")
            if message.get("name") not in (None, calls[call_id]):
                raise ReadinessError("tool result name differs from its call")
            pending.remove(call_id)
            results.append(call_id)
    terminal_action = route_name(record.get("route", "")) in {"tool_call", "tool_parallel", "tool_schema"}
    if pending and not (terminal_action and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls")):
        raise ReadinessError("complete training rows must contain every tool result")
    if selected == 0:
        raise ReadinessError("row has no supervised assistant message")
    return {"tools": sorted(known), "tool_call_ids": list(calls), "tool_result_ids": results,
            "supervised_messages": selected, "terminal_open_tool_calls": sorted(pending)}


def validate_media(record: dict[str, Any], root: Path, *, verify_files: bool = True) -> list[dict[str, Any]]:
    media = record.get("media", [])
    targets, inputs, out = set(), set(), []
    for asset in media:
        role = asset.get("role")
        if role not in {"input", "target", "conditioning"}:
            raise ReadinessError("media role must separate input, target and conditioning")
        source_sha = require_sha(asset.get("sha256"), "media.sha256")
        path = Path(asset.get("path", ""))
        path = path if path.is_absolute() else root / path
        if verify_files and (not path.is_file() or sha(path) != source_sha):
            raise ReadinessError(f"media missing or digest moved: {path}")
        if role == "target":
            targets.add(source_sha)
        else:
            inputs.add(source_sha)
        if asset.get("contains_clean_target") is True:
            raise ReadinessError("target leakage: clean target inside model input features")
        if asset.get("representation") == "latent" and role == "input" and ROUTES[route_name(record["route"])][0] == "D3b":
            sigma = asset.get("noise_level")
            if not isinstance(sigma, (int, float)) or not 0 < sigma <= 1:
                raise ReadinessError("flow input must be noisy with 0 < noise_level <= 1")
        out.append({**asset, "path": str(path.resolve())})
    if targets & inputs:
        raise ReadinessError("target leakage: same media bytes serve as target and input/conditioning")
    return out


def validate_record(record: dict[str, Any], root: Path, *, verify_files: bool = True) -> dict[str, Any]:
    if record.get("schema") != "recovery_corpus.v1":
        raise ReadinessError("record schema must be recovery_corpus.v1")
    route = route_name(record.get("route", ""))
    chat = validate_messages(record)
    if route.startswith("tool_") and not record.get("tools"):
        raise ReadinessError("tool route requires native tool schema context")
    media = validate_media(record, root, verify_files=verify_files)
    if ROUTES[route][2] and not media:
        raise ReadinessError(f"{route}: media route cannot train without real assets")
    return {"route": route, "chat": chat, "media": media}


@dataclass
class PackedRow:
    row_id: str
    route: str
    input_ids: list[int]
    assistant_mask: list[bool]
    next_token_mask: list[bool]
    template_sha256: str
    rendered_sha256: str
    linkage: dict[str, Any]
    source_record_sha256: str
    native_messages_sha256: str
    template_adaptations: list[dict[str, Any]]


def native_token_ids(value: Any) -> list[int]:
    """Extract one native conversation's IDs without changing token content.

    The current actual Qwen tokenizer returns BatchEncoding even for native
    apply_chat_template(tokenize=True). Iterating that object yields key names.
    List/tensor and one-row batch containers are accepted, multiple rows refuse.
    """
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ReadinessError("native tokenization mapping lacks input_ids")
        value = value["input_ids"]
    if callable(getattr(value, "tolist", None)):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list) or any(type(token) is not int for token in value):
        raise ReadinessError("native tokenization must contain exactly one integer-ID row")
    return value


def qwen_native_messages(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Adapt valid OpenAI JSON-string arguments to Qwen's native mapping input.

    Qwen's actual template iterates argument.items(). Both render and tokenize
    must receive the same copied native messages. Original records, schemas,
    call IDs and result IDs remain unchanged and separately digest-pinned.
    """
    messages = copy.deepcopy(record["messages"])
    adaptations = []
    for message in messages:
        for call in message.get("tool_calls", []):
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                decoded = json.loads(arguments)
                if not isinstance(decoded, dict):
                    raise ReadinessError("Qwen native tool arguments must be a mapping")
                call["function"]["arguments"] = decoded
                adaptations.append({"kind": "qwen_json_arguments_to_mapping", "tool_call_id": call["id"],
                                    "source_arguments_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
                                    "native_arguments_sha256": digest(decoded)})
    return messages, adaptations


def pack_native_chat(record: dict[str, Any], tokenizer: Any, *, root: Path, max_length: int) -> PackedRow:
    """Call the real parent template, then identify assistant spans in that output.

    No synthetic chat format and no substring search for answer text. ChatML is
    the audited Qwen parent format. Other templates refuse until an adapter is
    independently tested. Call/result ids remain intact in linkage metadata even
    when the parent's own template intentionally omits them from model text.
    """
    checked = validate_record(record, root)
    if checked["media"]:
        raise ReadinessError("text shard preparation cannot erase media; use the native media cache adapter")
    messages, adaptations = qwen_native_messages(record)
    options = {"tools": record.get("tools") or None, "add_generation_prompt": False, "enable_thinking": False}
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    if not isinstance(rendered, str):
        raise ReadinessError("chat template did not render text")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, (str, dict)) or not template:
        raise ReadinessError("parent tokenizer lacks a pinned chat template")
    spans = []
    assistant = [m for m, selected in zip(record["messages"], record["supervision"]["message_mask"])
                 if m["role"] == "assistant"]
    choices = [selected for m, selected in zip(record["messages"], record["supervision"]["message_mask"])
               if m["role"] == "assistant"]
    native_payload = canonical({"messages": record["messages"], "tools": record.get("tools", [])}).decode()
    if "<|im_start|>" in native_payload or "<|im_end|>" in native_payload:
        raise ReadinessError("unescaped role delimiters in content are ambiguous")
    headers = list(re.finditer(r"<\|im_start\|>assistant\n", rendered))
    if len(headers) != len(assistant):
        raise ReadinessError("native template assistant boundary count differs from messages")
    for header, selected in zip(headers, choices):
        end = rendered.find("<|im_end|>", header.end())
        if end < 0:
            raise ReadinessError("unterminated assistant turn in native template")
        if selected:
            spans.append((header.end(), end + len("<|im_end|>")))
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = native_token_ids(encoded), list(encoded["offset_mapping"])
    exact = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)
    if ids != native_token_ids(exact) or len(ids) != len(offsets):
        raise ReadinessError("native template ids disagree with offset-tokenized rendered output")
    if len(ids) > max_length:
        raise ReadinessError("row exceeds max_length; truncation could sever tool call/result linkage")
    mask = []
    for token, (begin, end) in zip(ids, offsets):
        if type(token) is not int or not 0 <= token < TEXT_VOCAB:
            raise ReadinessError("text token lies in ledger/music band or outside base vocabulary")
        if not 0 <= begin < end <= len(rendered):
            raise ReadinessError("token offsets must provide real character evidence")
        overlap = any(begin < hi and end > lo for lo, hi in spans)
        contained = any(lo <= begin and end <= hi for lo, hi in spans)
        if overlap != contained:
            raise ReadinessError("token crosses supervised/unsupervised template boundary")
        mask.append(contained)
    if not any(mask):
        raise ReadinessError("native template produced no supervised assistant token")
    next_mask = mask[1:] + [False]
    return PackedRow(str(record.get("id", record.get("record_id", ""))), checked["route"], ids, mask,
                     next_mask, digest(template), hashlib.sha256(rendered.encode()).hexdigest(), checked["chat"],
                     digest(record), digest(messages), adaptations)


def mask_streamed_targets(targets: dict[str, Any], assistant_mask: Any, lengths: Any) -> dict[str, Any]:
    """Apply assistant token ownership to the existing streamed trainer's tensors.

    Input mask is [B,L] token ownership; objectives at t predict token t+1.
    The mask covers ALL objective branches, not CE alone. Import torch only here.
    """
    import torch
    if assistant_mask.ndim != 2 or assistant_mask.dtype != torch.bool:
        raise ReadinessError("assistant_mask must be bool [B,L]")
    b, length = assistant_mask.shape
    if lengths.shape != (b,) or bool((lengths < 1).any()) or bool((lengths > length).any()):
        raise ReadinessError("lengths must be real in-row lengths")
    positions = torch.arange(length, device=assistant_mask.device)[None, :]
    if bool((assistant_mask & (positions >= lengths[:, None])).any()):
        raise ReadinessError("assistant mask supervises padding")
    predicting = torch.zeros_like(assistant_mask)
    predicting[:, :-1] = assistant_mask[:, 1:]
    predicting &= (positions + 1) < lengths[:, None]
    flat = predicting.reshape(-1)
    out = dict(targets)
    for name in ("ce_w", "eos_w", "q_w", "k_w"):
        if name in out:
            if out[name].numel() != b * length:
                raise ReadinessError(f"{name}: shape differs from token mask")
            out[name] = out[name] * flat.to(device=out[name].device, dtype=out[name].dtype)
    if "next_ids" in out:
        out["next_ids"] = torch.where(flat.to(out["next_ids"].device), out["next_ids"], -1)
    if "ul_ids" in out:
        out["ul_ids"] = torch.where(flat.to(out["ul_ids"].device)[:, None], out["ul_ids"], -1)
    if "jsd" in out:
        out["jsd"] = out["jsd"] & flat.to(out["jsd"].device)
    out["supervised_positions"] = flat
    return out


def optimizer_ownership(parameters: dict[str, Any], optimizer: Any) -> dict[str, Any]:
    """Prove each declared trainable leaf is owned exactly once by the optimizer.

    Pass the actual bank/delta tensors; a model name inventory alone cannot see
    D3's functional deltas outside the frozen module tree. No optimizer step runs.
    """
    declared = {}
    for name, tensor in parameters.items():
        if not tensor.requires_grad or not tensor.is_leaf:
            raise ReadinessError(f"{name}: declared trainable tensor must be a trainable leaf")
        if id(tensor) in declared:
            raise ReadinessError("aliased trainable tensor has multiple declared names")
        declared[id(tensor)] = name
    if not declared:
        raise ReadinessError("empty trainable census")
    owned = []
    for group in optimizer.param_groups:
        owned.extend(id(tensor) for tensor in group["params"])
    if len(owned) != len(set(owned)):
        raise ReadinessError("optimizer owns a trainable tensor more than once")
    if set(owned) != set(declared):
        raise ReadinessError("optimizer ownership differs from exact trainable census")
    inventory = [{"name": name, "shape": list(parameters[name].shape), "dtype": str(parameters[name].dtype),
                  "numel": parameters[name].numel()} for name in sorted(parameters)]
    return {"schema": "chimera_recovery.optimizer_ownership.v1", "tensors": inventory,
            "numel": sum(item["numel"] for item in inventory), "census_sha256": digest(inventory),
            "optimizer_step_performed": False}


def validate_native_batch(route: str, batch: dict[str, Any]) -> dict[str, Any]:
    """Validate the actual Chimera native bridge tensor names on CPU or device.

    This is executable tensor validation, not a production cache adapter. Native
    geometry and teacher provenance must be supplied by the adapter separately.
    """
    import torch
    name = route_name(route)
    stage, objective, allowed = ROUTES[name]
    if not allowed:
        raise ReadinessError("text-only route has no native media tensor contract")
    ids = batch["input_ids"]
    features = batch["native_media_features"]
    if ids.ndim != 2 or ids.dtype != torch.int64 or features.ndim != 3 or features.shape[0] != ids.shape[0] or features.shape[2] != FEATURES:
        raise ReadinessError("native batch requires int64 ids [B,L] and features [B,M,6144]")
    b, media, _ = features.shape
    shape = (b, media)
    types, indices = batch["native_media_type_ids"], batch["native_media_indices"]
    valid = batch["native_media_valid_mask"]
    positions = batch["native_media_positions"]
    if types.shape != shape or indices.shape != shape or types.dtype != torch.int64 or indices.dtype != torch.int64:
        raise ReadinessError("native types and indices must be int64 [B,M]")
    if valid.shape != shape or valid.dtype != torch.bool or positions.shape != (b, media, POSITIONS):
        raise ReadinessError("native validity must be bool [B,M] and positions [B,M,4]")
    if not bool(valid.any()) or bool(((indices < 0) | (indices >= ids.shape[1]))[valid].any()):
        raise ReadinessError("valid native rows require real in-sequence media indices")
    if bool((ids < 0).any()) or bool((ids >= TOTAL_VOCAB).any()):
        raise ReadinessError("native batch contains an out-of-vocabulary id")
    if any(int(kind) not in allowed for kind in torch.unique(types[valid]).tolist()):
        raise ReadinessError("native type dispatch differs from the route")
    if not bool(torch.isfinite(features[valid]).all()) or not bool(torch.isfinite(positions[valid]).all()):
        raise ReadinessError("non-finite native input or position")
    if stage == "D3b":
        target, weights, sigma = batch["native_media_targets"], batch["native_media_loss_mask"], batch["native_media_noise_levels"]
        if target.shape != features.shape or target.requires_grad or weights.shape != shape or sigma.shape != shape:
            raise ReadinessError("flow targets must be detached [B,M,F], weights and sigma [B,M]")
        if target.data_ptr() == features.data_ptr():
            raise ReadinessError("target leakage: input features alias teacher targets")
        if not bool(torch.isfinite(target[valid]).all()) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ReadinessError("non-finite target or invalid flow loss weights")
        if bool((weights[~valid] != 0).any()) or not bool((weights > 0).any()):
            raise ReadinessError("loss weights supervise padding or no media row")
        if not bool(torch.isfinite(sigma[valid]).all()) or bool(((sigma <= 0) | (sigma > 1))[valid].any()):
            raise ReadinessError("flow training requires finite 0 < sigma <= 1")
        if name == "joint_av":
            source = batch.get("charted_source_positions")
            if source is None or source.ndim != 3 or source.shape[0] != b or source.shape[-1] != 3 or not bool(torch.isfinite(source).all()):
                raise ReadinessError("H3 requires authentic packed [B,S,3] source positions")
    return {"route": name, "stage": stage, "objective": objective, "shape": list(features.shape),
            "valid_media_rows": int(valid.sum()), "production_adapter_verified": False}


def native_velocity_loss(prediction: Any, target: Any, types: Any, weights: Any, bands: dict[int, tuple[int, int]], *,
                         prediction_kind: str, target_kind: str, x_t: Any = None, sigma: Any = None,
                         parent_signs: dict[int, float] | None = None) -> Any:
    """Type-balanced emission-band MSE in canonical velocity (x_t - x0)/sigma.

    Frozen D3b specifies guided-velocity matching. A current x0 emission head can
    satisfy that objective via explicit differentiable conversion; x0 endpoint
    MSE is a different objective and requires a dated amendment. Parent raw
    targets use audited x0 = x_t + sign*sigma*raw, so velocity = -sign*raw.
    """
    import torch
    if prediction.shape != target.shape or prediction.ndim != 3 or types.shape != prediction.shape[:2] or weights.shape != types.shape:
        raise ReadinessError("velocity objective shape mismatch")
    if target.requires_grad or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ReadinessError("teacher target must be detached and weights finite/nonnegative")
    active = weights > 0
    if not bool(active.any()):
        raise ReadinessError("velocity objective has no supervised media rows")
    def convert(tensor, kind, rows, lo, hi, modality):
        value = tensor[..., lo:hi][rows].float()
        if kind == "canonical_velocity":
            return value
        if kind == "parent_raw":
            if parent_signs is None or parent_signs.get(modality) not in (-1, 1):
                raise ReadinessError("parent raw velocity requires an audited per-type sign")
            return -float(parent_signs[modality]) * value
        if kind == "x0":
            if x_t is None or sigma is None or x_t.shape != prediction.shape or sigma.shape != types.shape:
                raise ReadinessError("x0-to-velocity conversion requires exact noisy input and sigma")
            scale = sigma[rows].float()
            if not bool(torch.isfinite(scale).all()) or bool(((scale <= 0) | (scale > 1)).any()):
                raise ReadinessError("x0-to-velocity sigma must be finite in (0,1]")
            return (x_t[..., lo:hi][rows].float() - value) / scale[:, None]
        raise ReadinessError("explicit velocity/x0/parent_raw parameterization required")
    losses = []
    for modality in sorted(int(v) for v in torch.unique(types[active]).tolist()):
        if modality not in bands:
            raise ReadinessError("missing emission band for supervised modality")
        lo, hi = bands[modality]
        if not 0 <= lo < hi <= prediction.shape[-1]:
            raise ReadinessError("emission band outside native feature width")
        rows = (types == modality) & active
        pred = convert(prediction, prediction_kind, rows, lo, hi, modality)
        truth = convert(target, target_kind, rows, lo, hi, modality)
        if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(truth).all()):
            raise ReadinessError("velocity objective contains non-finite prediction or teacher")
        row_weights = weights[rows].float()
        losses.append(((pred - truth).square().mean(-1) * row_weights).sum() / row_weights.sum())
    return torch.stack(losses).mean()


def flow_velocity_loss(prediction: Any, x0_target: Any, types: Any, mask: Any, bands: dict[int, dict[str, Any]], *,
                       param: str = "x0", x_t: Any = None, sigma: Any = None,
                       signs: dict[int, float] | None = None, target_kind: str = "x0", x_t_layout: str = "feature",
                       noisy_feature_bands: dict[int, tuple[int, int]] | None = None) -> Any:
    """D3 flow_x0_loss-compatible hook, with a velocity objective.

    Python `param` names the emission head parameterization. It does not select
    the objective. Old raw-head predictions require signs; x0 heads require the
    exact noisy input and sigma, and always get sigma-aware velocity gradients.
    The default teacher argument is x0 to fit existing D3 teacher_x0 callers.
    """
    if param not in {"x0", "raw", "canonical_velocity"}:
        raise ReadinessError("unsupported D3 head parameterization")
    emission = {int(kind): tuple(band["emission"]) for kind, band in bands.items()}
    if x_t is not None and x_t_layout == "feature":
        # Bridge inputs and head outputs have distinct allocation tables. Never
        # assume a noisy latent occupies the same channels in both layouts.
        import torch
        aligned = torch.zeros_like(prediction)
        for modality in (int(value) for value in torch.unique(types[mask > 0]).tolist()):
            if modality not in bands or "feature" not in bands[modality]:
                raise ReadinessError("velocity conversion requires the audited feature-to-emission allocation")
            if noisy_feature_bands is None or modality not in noisy_feature_bands:
                raise ReadinessError("velocity conversion requires explicit audited noisy-channel suballocation")
            flo, fhi = noisy_feature_bands[modality]
            bridge_lo, bridge_hi = bands[modality]["feature"]
            elo, ehi = emission[modality]
            if (type(flo) is not int or type(fhi) is not int or fhi - flo != ehi - elo or
                    not 0 <= bridge_lo <= flo < fhi <= bridge_hi <= x_t.shape[-1]):
                raise ReadinessError("noisy-channel suballocation outside bridge band or differs from emission width")
            rows = types == modality
            aligned[..., elo:ehi][rows] = x_t[..., flo:fhi][rows]
        x_t = aligned
    elif x_t_layout != "emission":
        if x_t is not None or x_t_layout != "feature":
            raise ReadinessError("noisy input layout must be explicit feature or emission")
    return native_velocity_loss(prediction, x0_target, types, mask, emission,
                                prediction_kind="parent_raw" if param == "raw" else param,
                                target_kind=target_kind, x_t=x_t, sigma=sigma, parent_signs=signs)


@dataclass(frozen=True)
class ReadyObjectiveAdapter:
    """Bind actual sealed-cache tensors to D3b's loss_fn(prediction) callback.

    This adapter is usable by a production driver but cannot launch that driver.
    A recipe receipt is mandatory; synthetic rate checks use the pure loss hook
    separately and never produce this production-oriented binding.
    """
    cache_receipt: dict[str, Any]

    def __post_init__(self):
        receipt = self.cache_receipt
        if receipt.get("schema") != "chimera_recovery.native_flow_cache.v1" or any(
                receipt.get(field) is not True for field in ("sealed", "all_finite", "all_structural_ok", "guided_teacher", "no_clean_target_in_inputs")):
            raise ReadinessError("velocity adapter requires a sealed guided native cache and reviewed input/target separation")
        for field in ("cache_sha256", "corpus_sha256", "teacher_sha256", "recovery_amendment_sha256"):
            require_sha(receipt.get(field), field)
        if receipt.get("target_kind") not in {"x0", "canonical_velocity", "parent_raw"}:
            raise ReadinessError("sealed cache must declare teacher target parameterization")
        if receipt.get("prediction_kind") not in {"x0", "raw", "canonical_velocity"}:
            raise ReadinessError("sealed cache must declare student head parameterization")
        if receipt.get("input_layout") != "feature" or receipt.get("target_layout") != "emission":
            raise ReadinessError("sealed cache must declare native feature inputs and emission targets")
        allocation = receipt.get("noisy_feature_bands")
        if (not isinstance(allocation, dict) or not allocation or
                any(not isinstance(k, str) or not k.isdecimal() or not isinstance(v, list) or len(v) != 2 or
                    any(type(n) is not int for n in v) for k, v in allocation.items())):
            raise ReadinessError("sealed cache must pin explicit per-type noisy-channel suballocation")
        if digest(allocation) != require_sha(receipt.get("noisy_feature_bands_sha256"), "noisy_feature_bands_sha256"):
            raise ReadinessError("sealed noisy-channel suballocation digest moved")

    def bind(self, route: str, batch: dict[str, Any], bands: dict[int, dict[str, Any]], *,
             parent_signs: dict[int, float] | None = None):
        contract = validate_native_batch(route, batch)
        if contract["stage"] != "D3b":
            raise ReadinessError("velocity adapter can only bind a D3b flow route")
        if self.cache_receipt.get("route") != route_name(route):
            raise ReadinessError("sealed native cache route differs from objective binding")
        # Capture the exact adapter-provided tensors, not a recomputed/clean input.
        target = batch["native_media_targets"]
        types = batch["native_media_type_ids"]
        mask = batch["native_media_loss_mask"]
        x_t = batch["native_media_features"]
        sigma = batch["native_media_noise_levels"]
        allocation = {int(kind): tuple(value) for kind, value in self.cache_receipt["noisy_feature_bands"].items()}
        active_types = {int(value) for value in types[mask > 0].unique().tolist()}
        if not active_types.issubset(allocation):
            raise ReadinessError("sealed cache lacks supervised modality noisy-channel allocation")
        for kind in active_types:
            lo, hi = allocation[kind]
            flo, fhi = bands[kind]["feature"]
            elo, ehi = bands[kind]["emission"]
            if not 0 <= flo <= lo < hi <= fhi <= FEATURES or hi - lo != ehi - elo:
                raise ReadinessError("sealed noisy-channel allocation differs from native bridge/emission contract")
        if route_name(route) == "music_generate":
            # Audited ACE input is [source64, chunk_mask64, noisy64], each c*2+k.
            # Old S8 fixtures put x0 in source64 and must never become generation data.
            if (allocation.get(4) != (256, 384) or tuple(bands[4]["feature"]) != (0, 384) or
                    tuple(bands[4]["emission"]) != (0, 128)):
                raise ReadinessError("ACE generation requires audited noisy channels 256:384 -> emission 0:128")
            if self.cache_receipt.get("conditioning_contract") != {
                    "ace_source_latents": "zeros", "ace_chunk_masks": "zeros", "regime": "generation_without_target_reference"}:
                raise ReadinessError("ACE generation must declare target-free source and chunk-mask conditioning")
            if bool((x_t[..., :256][types == 4] != 0).any()):
                raise ReadinessError("ACE generation source/chunk conditioning contains nonzero target reference")
        def compute(prediction):
            return flow_velocity_loss(prediction, target, types, mask, bands,
                                      param=self.cache_receipt["prediction_kind"], target_kind=self.cache_receipt["target_kind"],
                                      x_t=x_t, sigma=sigma, signs=parent_signs, noisy_feature_bands=allocation)
        return compute


def d3_objective_patch(source: Path) -> tuple[str, dict[str, Any]]:
    """Add a real explicit CLI callback hook to D3's existing lr=0 rate driver.

    It leaves old x0 rate/synthetic defaults unchanged. An imported pure velocity
    adapter gets authentic noisy features/sigma from the segment. This is rate
    preparation only: rate targets remain labeled random and are not sealed-cache
    production supervision. ReadyObjectiveAdapter.bind supplies that later path.
    """
    baseline = sha(source)
    if baseline != D3_BASELINE_SHA256:
        raise ReadinessError("D3 baseline moved; inspect before regenerating objective patch")
    before = source.read_bytes().decode().replace("\r\n", "\n")
    after = before
    changes = [
        ('def rate_cell(hybrid: HybridD3, placement: UnitPlacement, seg: dict[str, Any], *, stage: str, steps: int) -> tuple[dict[str, Any], list[str]]:\n',
         'def rate_cell(hybrid: HybridD3, placement: UnitPlacement, seg: dict[str, Any], *, stage: str, steps: int, objective_adapter: Callable | None = None) -> tuple[dict[str, Any], list[str]]:\n'),
        ('        loss_fn = lambda pred: flow_x0_loss(pred, target, context["types"], mask, hybrid.bands)  # noqa: E731\n',
         '        if objective_adapter is None:\n'
         '            loss_fn = lambda pred: flow_x0_loss(pred, target, context["types"], mask, hybrid.bands)  # noqa: E731\n'
         '        else:\n'
         '            loss_fn = lambda pred: objective_adapter(pred, target, context["types"], mask, hybrid.bands, param="x0", x_t=seg["features"], sigma=seg["sigma"], signs=flow_signs())  # noqa: E731\n'
         '        cell.update({"flow_objective": "guided_velocity" if objective_adapter else "x0", "teacher_targets": "random rate fixture, not production cache"})\n'),
        ('def command_rate(args: argparse.Namespace) -> int:\n    started = time.time()\n',
         'def command_rate(args: argparse.Namespace) -> int:\n'
         '    objective_adapter = None\n'
         '    if args.flow_objective == "guided_velocity":\n'
         '        if not args.objective_adapter or not args.noisy_feature_bands:\n'
         '            raise D3Error("guided_velocity requires audited --objective-adapter and --noisy-feature-bands JSON")\n'
         '        adapter_path = Path(args.objective_adapter)\n'
         '        spec = importlib.util.spec_from_file_location("chimera_recovery_objective_adapter", adapter_path)\n'
         '        module = importlib.util.module_from_spec(spec)\n'
         '        sys.modules[spec.name] = module\n'
         '        spec.loader.exec_module(module)\n'
         '        adapter_fn = getattr(module, "flow_velocity_loss", None)\n'
         '        if not callable(adapter_fn):\n'
         '            raise D3Error("objective adapter lacks flow_velocity_loss")\n'
         '        allocation = json.loads(Path(args.noisy_feature_bands).read_text())\n'
         '        noisy_feature_bands = {int(k): tuple(v) for k, v in allocation.items()}\n'
         '        objective_adapter = lambda *a, **kw: adapter_fn(*a, **kw, noisy_feature_bands=noisy_feature_bands)\n'
         '    elif args.objective_adapter or args.noisy_feature_bands:\n'
         '        raise D3Error("adapter/allocation require --flow-objective guided_velocity")\n'
         '    started = time.time()\n'),
        ('        cell, cell_refusals = rate_cell(hybrid, placement, seg, stage=args.stage, steps=int(args.steps))\n',
         '        cell, cell_refusals = rate_cell(hybrid, placement, seg, stage=args.stage, steps=int(args.steps), objective_adapter=objective_adapter)\n'),
        ('    r.add_argument("--stage", default="d3b", choices=("d3b", "d3d"))\n',
         '    r.add_argument("--stage", default="d3b", choices=("d3b", "d3d"))\n'
         '    r.add_argument("--flow-objective", default="x0", choices=("x0", "guided_velocity"))\n'
         '    r.add_argument("--objective-adapter", help="audited recovery training.py exporting flow_velocity_loss; rate fixture only")\n'
         '    r.add_argument("--noisy-feature-bands", help="audited per-type noisy-channel suballocation JSON; required for velocity")\n'),
    ]
    for old, new in changes:
        if after.count(old) != 1:
            raise ReadinessError("D3 objective patch anchor absent or ambiguous")
        after = after.replace(old, new, 1)
    ast.parse(after)
    patch = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="a/d3_hybrid_module.py", tofile="b/d3_hybrid_module.py"))
    return patch, {"schema": "chimera_recovery.d3_objective_patch.v1", "baseline_sha256": baseline,
                   "patched_lf_sha256": hashlib.sha256(after.encode()).hexdigest(), "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                   "changes": len(changes), "objective_flag": "--flow-objective guided_velocity --objective-adapter <audited training.py> --noisy-feature-bands <audited allocation.json>",
                   "rate_only_random_teacher_targets": True, "production_driver_implemented": False, "amendment_required": True,
                   "frozen_preregistration_edited": False}


def streamed_supervision_patch(source: Path) -> tuple[str, dict[str, Any]]:
    """Emit a minimal patch against the audited exact campaign baseline.

    Corpus bytes, including assistant_mask, are already pinned by QTextCache.
    The new supervision flag is part of data-config identity and exact resume.
    Existing unmasked D2 data retains its frozen full-token behavior. Prepared
    assistant-mask data MUST explicitly select assistant_only in its config.
    """
    before_sha = sha(source)
    if before_sha != STREAMED_BASELINE_SHA256:
        raise ReadinessError("streamed trainer baseline moved; inspect it before regenerating the patch")
    raw = source.read_bytes()
    before = raw.decode("utf-8").replace("\r\n", "\n")
    after = before
    changes = [
        ('            keys = set(handle.keys())\n',
         '            keys = set(handle.keys())\n'
         '            self.corpus_assistant_mask = handle.get_tensor("assistant_mask") if "assistant_mask" in keys else None\n'),
        ('        self.seq_len = int(self.corpus_ids.shape[1])\n',
         '        self.seq_len = int(self.corpus_ids.shape[1])\n'
         '        if self.corpus_assistant_mask is not None:\n'
         '            if self.corpus_assistant_mask.dtype != torch.bool or self.corpus_assistant_mask.shape != self.corpus_ids.shape:\n'
         '                raise TrainerError("corpus assistant_mask must be bool and match input_ids")\n'),
        ('        return {"ids": ids, "length": length, "stored": stored, "topk_ids": topk_ids, "topk_logprobs": topk_lp}\n',
         '        assistant = None if self.corpus_assistant_mask is None else self.corpus_assistant_mask[row]\n'
         '        if assistant is not None and (bool(assistant[length:].any()) or not bool(assistant[:length].any())):\n'
         '            raise TrainerError("corpus assistant_mask supervises padding or no real assistant token")\n'
         '        return {"ids": ids, "length": length, "stored": stored, "topk_ids": topk_ids, "topk_logprobs": topk_lp, "assistant_mask": assistant}\n'),
        ('        xtok = None\n        row_ids = sorted(cache.index)\n',
         '        supervision = item.get("supervision")\n'
         '        if supervision not in (None, "all_tokens", "assistant_only"):\n'
         '            raise TrainerError("unknown supervision policy")\n'
         '        if (cache.corpus_assistant_mask is not None) != (supervision == "assistant_only"):\n'
         '            raise TrainerError("assistant-mask corpus and explicit assistant_only policy must agree")\n'
         '        xtok = None\n        row_ids = sorted(cache.index)\n'),
        ('                     "length_source": s.cache.length_source, "rows_cut_to_real_lengths": s.cache.rows_cut_to_real_lengths,\n',
         '                     "length_source": s.cache.length_source, "rows_cut_to_real_lengths": s.cache.rows_cut_to_real_lengths,\n'
         '                     "supervision": "assistant_only" if s.cache.corpus_assistant_mask is not None else "all_tokens",\n'),
        ('    return ids, targets\n\n\n# --------------------------------------------------------------------------\n# Optimisation plumbing',
         '    # Token t predicts t+1. Apply assistant ownership to ALL objective terms.\n'
         '    # Unmasked legacy sources retain their preregistered full-token objective.\n'
         '    supervised = torch.ones(batch, length, dtype=torch.bool)\n'
         '    for b, row in enumerate(rows):\n'
         '        if row.get("assistant_mask") is not None:\n'
         '            supervised[b].zero_()\n'
         '            supervised[b, :-1] = row["assistant_mask"][1:]\n'
         '            supervised[b] &= next_valid[b]\n'
         '    flat_supervised = supervised.reshape(-1)\n'
         '    for name in ("ce_w", "eos_w", "q_w", "k_w"):\n'
         '        if name in targets:\n'
         '            targets[name] = targets[name] * flat_supervised.to(targets[name].dtype)\n'
         '    targets["next_ids"] = torch.where(flat_supervised, targets["next_ids"], -1)\n'
         '    if "ul_ids" in targets:\n'
         '        targets["ul_ids"] = torch.where(flat_supervised[:, None], targets["ul_ids"], -1)\n'
         '    if "jsd" in targets:\n'
         '        targets["jsd"] &= flat_supervised\n'
         '    return ids, targets\n\n\n# --------------------------------------------------------------------------\n# Optimisation plumbing'),
    ]
    for old, new in changes:
        if after.count(old) != 1:
            raise ReadinessError("audited streamed trainer patch anchor absent or ambiguous")
        after = after.replace(old, new, 1)
    ast.parse(after)
    patch = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                      fromfile="a/streamed_kd_trainer.py", tofile="b/streamed_kd_trainer.py"))
    return patch, {"schema": "chimera_recovery.streamed_supervision_patch.v1", "baseline_sha256": before_sha,
                   "patched_lf_sha256": hashlib.sha256(after.encode()).hexdigest(), "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                   "changes": len(changes), "source": str(source.resolve()), "amendment_required": True,
                   "preserves_unmasked_legacy_objective": True, "frozen_preregistration_edited": False}


def audit_campaign(campaign: Path) -> dict[str, Any]:
    """Inspect pins and known execution gaps without importing production tools."""
    files = {}
    blockers = []
    for relative in PINNED_FILES:
        path = campaign / relative
        if path.is_file():
            files[relative] = {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}
        else:
            blockers.append(f"missing campaign source {relative}")
    prereg = campaign / PINNED_FILES[0]
    pinfile = campaign / "tools/stage2_training_receipt.py"
    pin = None
    if prereg.is_file() and pinfile.is_file():
        values = {}
        for node in ast.parse(pinfile.read_text(encoding="utf-8-sig")).body:
            if isinstance(node, ast.Assign):
                for name in node.targets:
                    if isinstance(name, ast.Name) and name.id.startswith("STAGE2_PREREGISTRATION_"):
                        try:
                            values[name.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
        raw = prereg.read_bytes()
        start = raw.find(values.get("STAGE2_PREREGISTRATION_SECTION", "## 1. Scope").encode())
        end = raw.find(values.get("STAGE2_PREREGISTRATION_NEXT_SECTION", "\n## Amendments").encode(), start)
        if start < 0 or end < 0:
            blockers.append("preregistration frozen-body markers absent")
        else:
            body = raw[start:end]
            observed = hashlib.sha256(body).hexdigest()
            pin = {"sha256": observed, "bytes": len(body), "expected_sha256": values.get("STAGE2_PREREGISTRATION_SHA256"),
                   "expected_bytes": values.get("STAGE2_PREREGISTRATION_BYTE_LENGTH"), "whole_document_sha256": sha(prereg)}
            pin["verified"] = observed == pin["expected_sha256"] and len(body) == pin["expected_bytes"]
            if not pin["verified"]:
                blockers.append("frozen preregistration body pin mismatch")
            d3 = campaign / "tools/d3_hybrid_module.py"
            if d3.is_file() and "guided-velocity KD" in body.decode() and 'default="x0"' in d3.read_text(encoding="utf-8-sig"):
                blockers.append("D3 x0 default conflicts with frozen guided-velocity objective; dated amendment required")
    return {"baseline_files": files, "preregistration": pin, "blockers": blockers}


def build_readiness(campaign: Path, records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    audit = audit_campaign(campaign)
    coverage = {name: 0 for name in ROUTES}
    invalid = []
    for i, record in enumerate(records):
        try:
            result = validate_record(record, root)
            coverage[result["route"]] += 1
        except (ReadinessError, KeyError, TypeError) as error:
            invalid.append({"row": i, "id": record.get("id"), "error": str(error)})
    lanes = {}
    for name, (stage, objective, types) in ROUTES.items():
        requirements = ["sealed recovery corpus + held split", "authentic parent-template assistant token masks",
                        "teacher cache digest + teacher identity", "registered immutable clone base",
                        "exact trainable census + optimizer ownership", "exact-resume receipt + finite gradients",
                        "dated recovery objective/data amendment", "native parity + assigned runtime resources"]
        if types:
            requirements.extend(["production media cache adapter without clean-target leakage", "D3 production driver + safety/resume/delta export"])
        if name == "audio_understand":
            requirements.append("audio-to-text conditioning/answer teacher and native adapter (not supplied by QVL image path)")
        if name == "video_understand":
            requirements.append("video frame/time alignment adapter and answer cache")
        lanes[name] = {"stage": stage, "objective": objective, "native_types": list(types),
                       "valid_rows": coverage[name], "production_status": "blocked", "unverified_requirements": requirements,
                       "preparation_status": "native_cache_adapter_missing" if types else "native_chat_packer_implemented",
                       "implemented_checks": ["route dispatch", "assistant message ownership", "tool schema/call/result validation", "media digest/target separation"]}
    return {"schema": SCHEMA, "training_started": False, "production_ready": False,
            "geometry": {"hidden": HIDDEN, "native_features": FEATURES, "positions": POSITIONS,
                         "text_vocab": [0, TEXT_VOCAB], "ledger": [TEXT_VOCAB, MUSIC_BEGIN],
                         "music_allocated": [MUSIC_BEGIN, TOTAL_VOCAB], "music_code_rows": MUSIC_CODES},
            "campaign": audit, "routes": lanes, "invalid_records": invalid,
            "scope": "CPU preparation and contract proof; no model or GPU job", "readiness_sha256": None}


def bind_text_cache(packed_manifest: Path, cache: Path, *, teacher_sha256: str) -> dict[str, Any]:
    """Return a usable D2 data config only after a matching teacher cache is sealed.

    The production QTextCache still re-verifies each shard and structural tensor
    contents; this step proves the mask-bearing corpus and requested teacher
    identity, and creates the explicit assistant_only policy it requires.
    """
    require_sha(teacher_sha256, "teacher_sha256")
    packed = json.loads(packed_manifest.read_text(encoding="utf-8"))
    if packed.get("schema") != PACK_SCHEMA:
        raise ReadinessError("packed manifest schema mismatch")
    if not packed.get("shard") or packed.get("rows", 1) < 1:
        raise ReadinessError("packed corpus contains no accepted rows")
    shard = Path(packed["shard"])
    if not shard.is_file() or sha(shard) != packed.get("sha256"):
        raise ReadinessError("prepared mask-bearing corpus digest moved")
    path = cache / "manifest.json"
    if not path.is_file():
        raise ReadinessError("teacher cache is not sealed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    teacher = manifest.get("teacher", {})
    if manifest.get("schema") != "omni-distill-teacher-cache-manifest-v1" or manifest.get("all_finite") is not True or manifest.get("all_structural_ok") is not True:
        raise ReadinessError("teacher cache schema, finite or structural gate failed")
    if teacher.get("equal_before_and_after_every_session") is not True or teacher.get("tree_sha256_before") != teacher_sha256 or teacher.get("tree_sha256_after") != teacher_sha256:
        raise ReadinessError("teacher cache identity differs from requested immutable teacher")
    if manifest.get("corpus", {}).get("kind") != "text" or manifest["corpus"].get("sha256") != packed["sha256"]:
        raise ReadinessError("teacher cache is not for the exact mask-bearing text corpus")
    if int(manifest.get("cache_config", {}).get("vocab_rows", 0)) != TEXT_VOCAB:
        raise ReadinessError("teacher distribution must use the exact text vocabulary band")
    for entry in manifest.get("shards", []):
        relative = Path(entry["shard"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ReadinessError("teacher shard path escapes its cache")
        info = cache / relative / "shard.json"
        if not info.is_file() or sha(info) != entry.get("shard_json_sha256"):
            raise ReadinessError("sealed teacher shard manifest missing or digest moved")
    if not manifest.get("shards"):
        raise ReadinessError("empty sealed teacher cache")
    return {"sources": [{"name": "recovery-assistant-text", "kind": "text", "qtext_cache": str(cache.resolve()),
                         "corpus": str(shard.resolve()), "supervision": "assistant_only", "weight": 1.0}],
            "recovery_binding": {"schema": "chimera_recovery.text_cache_binding.v1", "packed_manifest_sha256": sha(packed_manifest),
                                 "teacher_manifest_sha256": sha(path), "teacher_sha256": teacher_sha256,
                                 "production_training_ready": False}}


def prepare_text(records: list[dict[str, Any]], tokenizer: Any, root: Path, out: Path, max_length: int, *, split: str = "train") -> dict[str, Any]:
    """Write digest-pinned native text shards for qtext_teacher_cache.py.

    One complete conversation per row; no truncation, no separator packing that
    loses tool linkage, and no supervision on system/user/tool messages.
    """
    import torch
    from safetensors.torch import save_file
    if out.exists():
        raise ReadinessError("output already exists; preserve prepared artifacts")
    split = "validation" if split == "val" else split
    if split not in {"train", "validation", "test"}:
        raise ReadinessError("split must be train, validation or test")
    rows, accepted, quarantine, excluded = [], [], [], []
    groups = {}
    for record in records:
        record_split = "validation" if record.get("split") == "val" else record.get("split")
        group = source_group(record)
        if not isinstance(group, str) or not group or record_split not in {"train", "validation", "test"}:
            quarantine.append({"id": record.get("id"), "split": record_split, "reason": "missing stable group identity or valid split"})
            continue
        groups.setdefault(group, set()).add(record_split)
    overlapping = {group for group, splits in groups.items() if len(splits) != 1}
    for record in records:
        record_split = "validation" if record.get("split") == "val" else record.get("split")
        group = source_group(record)
        if not isinstance(group, str) or not group or record_split not in {"train", "validation", "test"}:
            continue
        if group in overlapping:
            quarantine.append({"id": record.get("id"), "split": record_split, "group": group, "reason": "source group occurs across splits"})
            continue
        if record_split != split:
            excluded.append({"id": record.get("id"), "split": record_split, "group": group})
            continue
        try:
            if record.get("provenance", {}).get("split", record_split) != record_split:
                raise ReadinessError("provenance split differs from row split")
            if record.get("quality", {}).get("train_ready") is False:
                raise ReadinessError("quality.train_ready explicitly false")
            rows.append(pack_native_chat(record, tokenizer, root=root, max_length=max_length))
            accepted.append(record)
        except Exception as error:
            if isinstance(error, MemoryError):
                raise
            quarantine.append({"id": record.get("id"), "route": record.get("route"), "split": record_split,
                               "group": group, "reason": str(error), "error_type": type(error).__name__})
    pad = getattr(tokenizer, "pad_token_id", None)
    if type(pad) is not int or not 0 <= pad < TEXT_VOCAB:
        raise ReadinessError("parent tokenizer must supply an in-text-vocabulary padding id")
    width = max((len(row.input_ids) for row in rows), default=0)
    ids = torch.full((len(rows), width), pad, dtype=torch.int64)
    masks = torch.zeros((len(rows), width), dtype=torch.bool)
    lengths = torch.tensor([len(row.input_ids) for row in rows], dtype=torch.int64)
    for i, row in enumerate(rows):
        ids[i, :len(row.input_ids)] = torch.tensor(row.input_ids)
        masks[i, :len(row.input_ids)] = torch.tensor(row.assistant_mask)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".recovery-pack-", dir=out.parent))
    tensor_path = out / "assistant_text.safetensors"
    staged_tensor = stage / tensor_path.name
    if rows:
        # QTextCache consumes target-indexed target_mask and shifts it internally.
        # Separate storage avoids safetensors rejecting aliased tensor names.
        save_file({"input_ids": ids, "lengths": lengths, "assistant_mask": masks, "target_mask": masks.clone()},
                  str(staged_tensor), metadata={"schema": PACK_SCHEMA})
    manifest = {"schema": PACK_SCHEMA, "corpus_schema": "recovery_corpus.v1", "shard": str(tensor_path.resolve()) if rows else None,
                "sha256": sha(staged_tensor) if rows else None, "rows": len(rows), "tokens": int(lengths.sum()),
                "supervised_tokens": int(masks.sum()), "shape": list(ids.shape),
                "split": split, "input_records_sha256": digest(records), "records_sha256": digest(accepted),
                "template_sha256": rows[0].template_sha256 if rows else digest(getattr(tokenizer, "chat_template", None)),
                "quarantined_rows": quarantine, "quarantined_count": len(quarantine), "other_split_rows_preserved": excluded,
                "other_split_count": len(excluded), "cross_split_groups": sorted(overlapping),
                "admission": "accepted rows only; holdouts require distinct --split outputs; no truncated conversations",
                "row_receipts": [{**asdict(row), "input_ids": None, "assistant_mask": None, "next_token_mask": None} for row in rows],
                "teacher_cache_required": True, "production_training_ready": False,
                "trainer_integration": "apply streamed_supervision.patch against its recorded baseline sha; corpus assistant_mask is token ownership"}
    (stage / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    stage.rename(out)
    return manifest


def cpu_dryrun() -> dict[str, Any]:
    """Exercise every route's real batch shape/loss ownership contract on CPU."""
    checks = []
    for route, (stage, objective, types) in ROUTES.items():
        contract = {"route": route, "stage": stage, "objective": objective, "input_ids": [1, 16],
                    "lengths": [1], "assistant_mask": [1, 16], "next_token_mask": [1, 16],
                    "text_head": [TEXT_VOCAB, HIDDEN], "supervised_roles": ["assistant"]}
        if types:
            contract.update({"native_media_features": [1, 8, FEATURES], "native_media_type_ids": [1, 8],
                             "native_media_positions": [1, 8, POSITIONS], "native_media_indices": [1, 8],
                             "native_media_valid_mask": [1, 8], "native_media_targets": [1, 8, FEATURES],
                             "native_media_noise_levels": [1, 8], "native_media_loss_mask": [1, 8],
                             "native_type_ids": list(types), "targets_are_inputs": False})
        if route == "joint_av":
            contract["h3_source_positions"] = [1, 24, 3]
            contract["coupled_sigma_schedule"] = {"video_shift": 12, "audio_shift": 3}
        checks.append(contract)
    return {"schema": "chimera_recovery.cpu_contract_dryrun.v1", "route_contracts": checks,
            "passed": len(checks), "model_forward_performed": False, "backward_performed": False,
            "proof_limit": "shape and ownership contract only; native numerical parity is an external prerequisite"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("cpu-dryrun")
    dry.add_argument("--out", type=Path, required=True)
    ready = commands.add_parser("readiness")
    ready.add_argument("--campaign", type=Path, required=True)
    ready.add_argument("--corpus", type=Path)
    ready.add_argument("--out", type=Path, required=True)
    prepare = commands.add_parser("prepare-text")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--tokenizer", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--max-length", type=int, default=4096)
    prepare.add_argument("--split", default="train", choices=("train", "validation", "val", "test"))
    patch = commands.add_parser("patch-streamed")
    patch.add_argument("--source", type=Path, required=True)
    patch.add_argument("--out", type=Path, required=True)
    d3patch = commands.add_parser("patch-d3-objective")
    d3patch.add_argument("--source", type=Path, required=True)
    d3patch.add_argument("--out", type=Path, required=True)
    bind = commands.add_parser("bind-text-cache")
    bind.add_argument("--packed-manifest", type=Path, required=True)
    bind.add_argument("--cache", type=Path, required=True)
    bind.add_argument("--teacher-sha256", required=True)
    bind.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "bind-text-cache":
        config = bind_text_cache(args.packed_manifest, args.cache, teacher_sha256=args.teacher_sha256)
        if args.out.exists():
            raise ReadinessError("preserve existing data config; choose a new output")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(canonical(config) + b"\n")
        print(json.dumps(config["recovery_binding"]))
        return 0
    if args.command in {"patch-streamed", "patch-d3-objective"}:
        patch_text, receipt = (streamed_supervision_patch if args.command == "patch-streamed" else d3_objective_patch)(args.source)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(patch_text.encode("utf-8"))
        args.out.with_suffix(args.out.suffix + ".json").write_bytes(canonical(receipt) + b"\n")
        print(json.dumps(receipt))
        return 0
    if args.command == "cpu-dryrun":
        result = cpu_dryrun()
    else:
        records = [] if not args.corpus else [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.command == "readiness":
            result = build_readiness(args.campaign, records, args.corpus.parent if args.corpus else Path.cwd())
            result["readiness_sha256"] = digest({k: v for k, v in result.items() if k != "readiness_sha256"})
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
            manifest = prepare_text(records, tokenizer, args.corpus.parent, args.out, args.max_length, split=args.split)
            print(json.dumps({"schema": manifest["schema"], "rows": manifest["rows"], "supervised_tokens": manifest["supervised_tokens"],
                              "split": manifest["split"], "quarantined": manifest["quarantined_count"], "other_split_count": manifest["other_split_count"],
                              "production_training_ready": False}))
            return 0 if manifest["rows"] else 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical(result) + b"\n")
    print(json.dumps({"schema": result["schema"], "out": str(args.out), "production_ready": result.get("production_ready", False)}))
    return 0 if args.command == "cpu-dryrun" else 3


if __name__ == "__main__":
    raise SystemExit(main())
