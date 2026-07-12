#!/usr/bin/env python3
"""Run a bounded parent/codebook cached residual-attention probe.

This is an evaluation-only bridge between the live Qwen cached trace collector
and the residual-attention scorer.  It deliberately loads the dense reference
and compact runtime sequentially, keeps only detached CPU traces, and never
edits either model.  Parent vocabulary rows are projected into the compact
runtime's exact tokenizer chart before evidence is scored.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_forge.hf_cached_attention_trace import (
    HFCachedAttentionStep,
    HFCachedAttentionTrace,
    HFCachedAttentionTraceLimits,
    build_residual_attention_candidate_evidence,
    collect_hf_cached_attention_trace,
)
from neural_forge.hf_qwen_codebook_model import load_qwen3_codebook_checkpoint
from neural_forge.hf_residual_attention_fold import (
    ResidualAttentionScoringConfig,
    score_residual_attention_candidate,
)


SCHEMA = "neural-forge-qwen-cached-residual-attention-probe-v1"


class ProbeError(RuntimeError):
    """A probe input, checkpoint, or measured invariant failed closed."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--codebook-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--continuation", action="append", required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--group", type=int)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-capture-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--logit-rtol", type=float, default=5e-3)
    parser.add_argument("--logit-atol", type=float, default=5e-3)
    parser.add_argument("--context-rtol", type=float, default=8e-3)
    parser.add_argument("--context-atol", type=float, default=8e-3)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _state_tensor(checkpoint: Path, suffix: str) -> torch.Tensor:
    tensor_path = checkpoint / "model.safetensors"
    if not tensor_path.is_file():
        raise ProbeError(f"Missing canonical tensor file: {tensor_path}")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        matches = [key for key in handle.keys() if key == suffix or key.endswith(f".{suffix}")]
        if len(matches) != 1:
            raise ProbeError(
                f"Checkpoint must contain exactly one {suffix!r} tensor; found {matches}."
            )
        return handle.get_tensor(matches[0]).detach().cpu().clone()


def _affected_site(
    assignments: torch.Tensor,
    requested_layer: int | None,
    requested_group: int | None,
) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    if assignments.ndim != 2 or assignments.numel() == 0:
        raise ProbeError("Attention assignments must be a non-empty [layers, groups] tensor.")
    assignments = assignments.to(dtype=torch.int64)
    layers, groups = map(int, assignments.shape)
    shared: list[tuple[int, int, int]] = []
    for group in range(groups):
        values = assignments[:, group]
        for slot in values.unique(sorted=True).tolist():
            members = torch.nonzero(values == int(slot), as_tuple=False).flatten().tolist()
            if len(members) > 1:
                shared.extend((int(layer), group, int(slot)) for layer in members)
    if requested_layer is None and requested_group is None:
        if not shared:
            raise ProbeError("Checkpoint has no shared attention prototype to probe.")
        layer, group, slot = shared[0]
        slot_sites = tuple(
            (member_layer, group)
            for member_layer in range(layers)
            if int(assignments[member_layer, group]) == slot
        )
        return layer, group, slot, slot_sites
    if requested_layer is None or requested_group is None:
        raise ProbeError("--layer and --group must be supplied together.")
    if not 0 <= requested_layer < layers or not 0 <= requested_group < groups:
        raise ProbeError("Requested layer/group is outside the attention assignment table.")
    slot = int(assignments[requested_layer, requested_group])
    slot_sites = tuple(
        (member_layer, requested_group)
        for member_layer in range(layers)
        if int(assignments[member_layer, requested_group]) == slot
    )
    return requested_layer, requested_group, slot, slot_sites


def _logical_cases(
    tokenizer: Any,
    prompts: list[str],
    continuations: list[str],
    horizon: int,
    max_prompt_tokens: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if len(prompts) != len(continuations):
        raise ProbeError("--prompt and --continuation counts must match.")
    if horizon <= 0 or horizon > 32:
        raise ProbeError("horizon must be in [1, 32].")
    cases: list[tuple[torch.Tensor, torch.Tensor]] = []
    for prompt, continuation in zip(prompts, continuations):
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
        continuation_ids = tokenizer(
            continuation,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        prompt_length = int(prompt_ids.shape[1])
        if prompt_length <= 0 or prompt_length > max_prompt_tokens:
            raise ProbeError("A prompt is empty or exceeds max_prompt_tokens.")
        if int(continuation_ids.shape[1]) < horizon:
            raise ProbeError(
                "Each continuation must supply at least horizon forced tokens."
            )
        forced = continuation_ids[0, :horizon].to(torch.long)
        cases.append((prompt_ids.to(torch.long), forced))
    return tuple(cases)


def _map_reference_trace(
    trace: HFCachedAttentionTrace,
    *,
    logical_prompt: torch.Tensor,
    logical_forced: torch.Tensor,
    parent_rows: torch.Tensor,
) -> HFCachedAttentionTrace:
    rows = parent_rows.to(dtype=torch.long)
    mapped_steps: list[HFCachedAttentionStep] = []
    for index, step in enumerate(trace.steps):
        logits = step.action_logits.index_select(-1, rows)
        mapped_steps.append(
            replace(
                step,
                selected_token_id=int(logical_forced[index]),
                action_logits=logits,
            )
        )
    return replace(
        trace,
        prompt_token_ids=tuple(int(value) for value in logical_prompt.reshape(-1)),
        consumed_token_ids=tuple(
            int(value) for value in logical_forced[:-1].reshape(-1)
        ),
        selected_token_ids=tuple(int(value) for value in logical_forced.reshape(-1)),
        steps=tuple(mapped_steps),
    )


def _release(model: torch.nn.Module) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = _parse_args()
    checkpoint = Path(args.codebook_checkpoint).resolve()
    output = Path(args.output).resolve()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ProbeError("A CUDA device was requested but CUDA is unavailable.")

    parent_rows = _state_tensor(checkpoint, "parent_token_rows").to(torch.long)
    assignments = _state_tensor(
        checkpoint, "attention_codebook.registry.assignments"
    ).to(torch.long)
    group_permutations = _state_tensor(
        checkpoint, "attention_codebook.registry.group_permutations"
    ).to(torch.long)
    prototype_sources = _state_tensor(
        checkpoint, "attention_codebook.registry.prototype_source_indices"
    ).to(torch.long)
    if group_permutations.shape != assignments.shape:
        raise ProbeError("Attention assignments and group permutations disagree.")
    layer, group, slot, shared_sites = _affected_site(
        assignments, args.layer, args.group
    )
    if (
        prototype_sources.ndim != 2
        or int(prototype_sources.shape[0]) != int(assignments.shape[1])
        or slot >= int(prototype_sources.shape[1])
    ):
        raise ProbeError("Attention prototype provenance has incompatible shape.")
    source_index = int(prototype_sources[group, slot])
    source_layer, source_physical_group = divmod(
        source_index, int(assignments.shape[1])
    )
    if args.layer is None:
        redirected = [site for site in shared_sites if site[0] != source_layer]
        if not redirected:
            raise ProbeError("Shared prototype has no redirected logical site to probe.")
        layer, group = redirected[0]
    physical_group = int(group_permutations[layer, group])
    if int(group_permutations[source_layer, group]) != source_physical_group:
        raise ProbeError("Prototype source provenance disagrees with group permutations.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.dense_model, local_files_only=True, trust_remote_code=False
    )
    if int(parent_rows.numel()) != len(tokenizer):
        raise ProbeError(
            "Compact parent-token row count does not equal the tokenizer vocabulary."
        )
    logical_cases = _logical_cases(
        tokenizer,
        list(args.prompt),
        list(args.continuation),
        args.horizon,
        args.max_prompt_tokens,
    )
    limits = HFCachedAttentionTraceLimits(
        max_prompts=len(logical_cases),
        max_horizon=args.horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_layers=1,
        max_groups=1,
        max_model_devices=1,
        max_capture_bytes=args.max_capture_bytes,
        logit_rtol=args.logit_rtol,
        logit_atol=args.logit_atol,
        context_rtol=args.context_rtol,
        context_atol=args.context_atol,
    )

    dense = AutoModelForCausalLM.from_pretrained(
        args.dense_model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map={"": args.device},
    ).eval()
    reference_traces: list[HFCachedAttentionTrace] = []
    for case_index, (logical_prompt, logical_forced) in enumerate(logical_cases):
        parent_prompt = parent_rows.index_select(0, logical_prompt.reshape(-1)).reshape_as(
            logical_prompt
        )
        parent_forced = parent_rows.index_select(0, logical_forced)
        raw_trace = collect_hf_cached_attention_trace(
            dense,
            parent_prompt,
            layer_index=layer,
            group_indices=(physical_group,),
            horizon=args.horizon,
            sequence_id=f"case-{case_index}",
            lane="forced",
            forced_token_ids=parent_forced,
            limits=limits,
        )
        reference_traces.append(
            _map_reference_trace(
                raw_trace,
                logical_prompt=logical_prompt,
                logical_forced=logical_forced,
                parent_rows=parent_rows,
            )
        )
    _release(dense)

    child = load_qwen3_codebook_checkpoint(checkpoint, device="cpu")
    child.to(dtype=dtype)
    child.to(args.device)
    child.eval()
    folded_traces: list[HFCachedAttentionTrace] = []
    for case_index, (logical_prompt, logical_forced) in enumerate(logical_cases):
        folded_traces.append(
            collect_hf_cached_attention_trace(
                child,
                logical_prompt,
                layer_index=layer,
                group_indices=(physical_group,),
                horizon=args.horizon,
                sequence_id=f"case-{case_index}",
                lane="forced",
                forced_token_ids=logical_forced,
                limits=limits,
            )
        )
    parameter_report = getattr(child, "neural_forge_codebook_report").as_dict()
    _release(child)

    evidence = build_residual_attention_candidate_evidence(
        reference_traces,
        folded_traces,
        candidate_ids=(
            f"layer-{layer}:canonical-group-{group}:physical-group-"
            f"{physical_group}:slot-{slot}",
        ),
    )
    damage = score_residual_attention_candidate(
        evidence, ResidualAttentionScoringConfig(top_k=5)
    )
    damage_payload = damage.as_dict()
    damage_payload["closed_loop"] = False
    damage_payload["cache_recurrent_transition"] = True
    damage_payload["token_lane"] = "forced-shared-logical-token-prefix"
    damage_payload["autoregressive_action_feedback_verified"] = False
    manifest_path = checkpoint / "manifest.json"
    payload = {
        "schema": SCHEMA,
        "dense_model": str(Path(args.dense_model).resolve()),
        "codebook_checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": _sha256_file(manifest_path),
        "evaluation_only": True,
        "training_used": False,
        "gradient_used": False,
        "mutation_used": False,
        "lane": "forced-shared-logical-token-prefix",
        "layer": layer,
        "canonical_group": group,
        "physical_group": physical_group,
        "prototype_slot": slot,
        "prototype_source_layer": source_layer,
        "prototype_source_physical_group": source_physical_group,
        "selected_site_role": (
            "survivor-source" if layer == source_layer else "redirected-source"
        ),
        "shared_assignment_sites": [list(value) for value in shared_sites],
        "prompt_count": len(logical_cases),
        "horizon": args.horizon,
        "dtype": str(dtype),
        "device": args.device,
        "numeric_tolerances": {
            "logit_rtol": args.logit_rtol,
            "logit_atol": args.logit_atol,
            "context_rtol": args.context_rtol,
            "context_atol": args.context_atol,
        },
        "parent_rows_identity": bool(
            torch.equal(parent_rows, torch.arange(parent_rows.numel()))
        ),
        "parent_trace_summaries": [trace.as_dict() for trace in reference_traces],
        "folded_trace_summaries": [trace.as_dict() for trace in folded_traces],
        "damage": damage_payload,
        "parameter_report": parameter_report,
    }
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
