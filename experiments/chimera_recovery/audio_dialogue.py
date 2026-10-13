"""Actual parent ChatML text with explicit native audio observation insertion.

Preserves all native parent text IDs and assistant body/EOS ownership. This
CPU packer provides no encoder, coordinates, training admission or fake audio
processor. Media rows go inside the user's turn before its native closing EOS.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_audio_dialogue_common", HERE / "training.py")
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)
Error = R.ReadinessError
PLACEHOLDER = 248832
POLICY = "native_audio_before_user_close_v1"
TOKENIZER_PINS = {
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}


def pack_dialogue(record, tokenizer, *, media_rows, max_length):
    """Native generation-prefix anchor plus lossless media-ID insertion."""
    if (record.get("route") not in {"audio_understand", "speech_understand"} or type(media_rows) is not int or media_rows < 1 or
            type(max_length) is not int or max_length < 1 or record.get("tools") or
            record.get("supervision", {}).get("policy") != "assistant_only" or
            record.get("supervision", {}).get("message_mask") != [False, True]):
        raise Error("audio dialogue requires one terminal assistant and explicit observation rows")
    messages = copy.deepcopy(record["messages"])
    if ([message.get("role") for message in messages] != ["user", "assistant"] or
            any(not isinstance(message.get("content"), str) or not message["content"] or message.get("tool_calls") for message in messages) or
            any(token in message["content"] for message in messages for token in ("<|im_start|>", "<|im_end|>"))):
        raise Error("audio dialogue needs unambiguous ordinary parent text turns; no erased media content")
    options = {"tools": None, "enable_thinking": False}
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **options)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **options)
    if not isinstance(full, str) or not isinstance(prompt, str) or not full.startswith(prompt):
        raise Error("native assistant generation prompt is not an exact reference prefix")
    full_ids = R.native_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, **options))
    prompt_ids = R.native_token_ids(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, **options))
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(encoded["offset_mapping"])
    if full_ids != R.native_token_ids(encoded) or full_ids[:len(prompt_ids)] != prompt_ids or len(offsets) != len(full_ids):
        raise Error("native template text IDs, offsets or generation-prefix IDs disagree")
    if any(type(token) is not int or not 0 <= token < R.TEXT_VOCAB for token in full_ids):
        raise Error("parent dialogue token outside authentic text vocabulary")
    headers = list(re.finditer(r"<\|im_start\|>user\n", full))
    if len(headers) != 1:
        raise Error("native template must expose one actual user close boundary")
    close = full.find("<|im_end|>", headers[0].end())
    anchors = [index for index, (begin, end) in enumerate(offsets) if begin == close and end == close + len("<|im_end|>")]
    eos_ids = R.native_token_ids(tokenizer("<|im_end|>", add_special_tokens=False))
    if len(anchors) != 1 or len(eos_ids) != 1 or full_ids[anchors[0]] != eos_ids[0] or anchors[0] >= len(prompt_ids):
        raise Error("user audio insertion lacks exact native closing-EOS token/offset evidence")
    boundary = anchors[0]
    end_candidates = [index for index in range(len(prompt_ids), len(full_ids)) if full_ids[index] == eos_ids[0]]
    if len(end_candidates) != 1 or end_candidates[0] <= len(prompt_ids):
        raise Error("assistant reference needs a nonempty body and one authentic completion EOS")
    assistant_end = end_candidates[0]
    if full[offsets[assistant_end][1]:].strip():
        raise Error("extra assistant turns or payload after terminal native EOS")
    base_mask = [len(prompt_ids) <= index <= assistant_end for index in range(len(full_ids))]
    for selected, (begin, end) in zip(base_mask, offsets):
        if not 0 <= begin < end <= len(full) or selected and begin < len(prompt):
            raise Error("native token offset crosses the assistant generation-prefix boundary")
    ids = full_ids[:boundary] + [PLACEHOLDER] * media_rows + full_ids[boundary:]
    mask = base_mask[:boundary] + [False] * media_rows + base_mask[boundary:]
    indices = list(range(boundary, boundary + media_rows))
    if len(ids) > max_length:
        raise Error("audio dialogue exceeds max_length; no waveform, native rows or answer truncation permitted")
    if [token for index, token in enumerate(ids) if index not in indices] != full_ids:
        raise Error("native media insertion changed parent text IDs")
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise Error("actual parent chat template must be pinned")
    return {"schema": "chimera_recovery.audio_dialogue.v1", "policy": POLICY,
            "input_ids": ids, "assistant_target_mask": mask, "next_token_mask": mask[1:] + [False],
            "media_indices": indices, "prompt_length": len(prompt_ids) + media_rows,
            "completion_eos_id": eos_ids[0], "assistant_targets": sum(mask),
            "parent_text_ids": full_ids, "native_text_projection_bitwise_equal": True,
            "template_sha256": R.digest(template), "rendered_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest(),
            "source_record_sha256": R.digest(record), "production_training_ready": False,
            "requires": ["real source-pinned native encoder/positions/admission", "dated audio insertion/read-policy objective amendment"]}


def inspect_source(source, preflight, preflight_sha, tokenizer_dir, out, *, max_length=4096):
    if out.exists():
        raise Error("preserve existing actual tokenizer inspection")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise Error("actual tokenizer inspection requires CUDA_VISIBLE_DEVICES='' and no model weights")
    if R.sha(preflight) != R.require_sha(preflight_sha, "source_preflight_sha256"):
        raise Error("source preflight changed")
    receipt = json.loads(preflight.read_text(encoding="utf-8"))
    if receipt.get("source_ready") is not True or receipt.get("source_sha256") != R.sha(source):
        raise Error("source rows differ from verified audio preflight")
    for filename, expected in TOKENIZER_PINS.items():
        if R.sha(tokenizer_dir / filename) != expected:
            raise Error("actual parent tokenizer source differs: " + filename)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
    checked = {row["id"]: row for row in receipt["rows"]}
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("route") != "audio_understand":
            continue
        known = checked.get(record["id"])
        if known is None or known["source_row_sha256"] != R.digest(record):
            raise Error("actual source row no longer matches verified audio preflight")
        frames = known["duration_seconds"] * 40
        if not math.isclose(frames, round(frames), abs_tol=1e-9):
            raise Error("tokenizer-only source inspection requires source duration on native40Hz lattice")
        packed = pack_dialogue(record, tokenizer, media_rows=2 * round(frames), max_length=max_length)
        rows.append({"id": record["id"], "split": known["split"], "recovery_group_id": known["recovery_group_id"],
                     "asset_sha256": known["waveform_sha256"], "encoded_media_rows_verified": False,
                     "dialogue": packed})
    if len(rows) != len(checked) or len({row["id"] for row in rows}) != len(rows):
        raise Error("native audio tokenizer inspection must cover every source row exactly once")
    result = {"schema": "chimera_recovery.audio_dialogue_inspection.v1", "inspection_passed": True,
              "production_training_ready": False, "model_weights_loaded": False,
              "source_sha256": R.sha(source), "source_preflight_sha256": preflight_sha,
              "tokenizer_files": TOKENIZER_PINS, "adapter_sha256": R.sha(Path(__file__)),
              "rows": rows, "policy": POLICY, "scope": "native text IDs and mask/insertion anchors; actual media encode/XYZ/admission still separate"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(R.canonical(result) + b"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-preflight", "tokenizer-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-preflight-sha256", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args(argv)
    result = inspect_source(args.source, args.source_preflight, args.source_preflight_sha256, args.tokenizer_dir, args.out, max_length=args.max_length)
    print(json.dumps({"inspection_passed": True, "rows": len(result["rows"]), "production_training_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
