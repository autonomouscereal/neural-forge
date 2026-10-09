#!/usr/bin/env python3
"""Measured dual-stream QIMG GELU channel folds; no installer or media proof.

The frozen FP8 parent is explicitly cast to its captured BF16 inference dtype.
Finite source replay and fidelity targets are diagnostics under the explicit
current owner policy; they never prevent selection or artifact publication. Only image-disjoint FIT rows enter the existing prior-anchored solver;
actual captured MLP outputs, not a recreated parent's outputs, are its targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parents[1] / "media"))
from c16_qimg_capture import CAPTURE_SCHEMA, SOURCE_FILES, PARENT_HEADER_SHA256, file_sha, tensor_sha, write_tensors
from c16_qimg_preflight import biased_gelu_mlp_parameters, maximum_fourfold_keep
from mediafold.linalg import law_abiding_solve, pivoted_columns_gram, write_scores

STREAMS = ("img", "txt")
POLICY_SCHEMA = "neural_forge.qimg_fold_policy.v1"
# Keep the historical quality-gated schema distinct; never relabel old receipts.
FOLD_SCHEMA = "neural_forge.qimg_dual_stream_fold.v1"
OWNER_FOLD_SCHEMA = "neural_forge.qimg_dual_stream_development_artifact.v2"
OWNER_POLICY_ID = "chimera_omni.owner_finite_diagnostics_only.2026-10-06.v1"


def current_owner_policy():
    return {"policy_id": OWNER_POLICY_ID, "finite_numeric_targets": "diagnostic_only",
            "artifact_authority": "structural validity only; unevaluated development artifact",
            "capability_evaluation": "after fold and assembly; no capability success inferred",
            "historical_policy": "caller-pinned reference values retained without rewriting past decisions",
            "solver_constraint": "unchanged law_abiding_solve filters update_scale>1 rungs and may retain prior; not an artifact validity gate"}


def current_artifact_fields():
    return {"owner_policy": current_owner_policy(), "structural_artifact_valid": True,
            "accepted": False, "acceptance_enabled": False, "capability_accepted": False,
            "artifact_state": "unevaluated_development", "numerical_targets_are_diagnostics": True,
            "diagnostic_target_scope": ["finite source replay reference", "pooled and each held-forward error/cosine references",
                "stored BF16 update<=1 reference", "bias-inclusive fourfold reference", "2*keep FIT-row conditioning reference",
                "historical selector full-rank selection without fallback", "derived diagnostic availability"]}


class DiagnosticMetadataError(ValueError):
    """Receipt metadata is not JSON data; no tensor sanitization is permitted."""


def json_diagnostics(value):
    """Represent undefined derived numbers as null with explicit field paths.

    This never sanitizes tensors or coefficients. Those must pass their own
    finite/shape checks before diagnostics are built.
    """
    unavailable = []
    def visit(item, path):
        if isinstance(item, float) and not math.isfinite(item):
            unavailable.append({"field": path, "classification": "undefined_or_nonfinite_derived_metric", "value": str(item)})
            return None
        if isinstance(item, dict):
            if not all(isinstance(k, str) for k in item): raise DiagnosticMetadataError("diagnostic keys must be strings")
            return {k: visit(v, path + "." + k if path else k) for k, v in item.items()}
        if isinstance(item, (tuple, list)): return [visit(v, f"{path}[{i}]") for i, v in enumerate(item)]
        if item is None or isinstance(item, (str, bool, int, float)): return item
        raise DiagnosticMetadataError("diagnostics must contain JSON metadata only, never tensors")
    result = visit(value, "")
    if unavailable:
        if not isinstance(result, dict): raise ValueError("diagnostic root must be a metadata object")
        result["undefined_diagnostics"] = unavailable
    return result


def finite_at_most(value, target):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value <= target


def diagnostics_complete(value):
    if isinstance(value, dict):
        return not value.get("undefined_diagnostics") and all(diagnostics_complete(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(diagnostics_complete(v) for v in value)
    return not (isinstance(value, float) and not math.isfinite(value))


def all_numeric_targets_met(result, stream_numeric_key):
    """Descriptive aggregation only; no outcome of this function vetoes tensors."""
    return (diagnostics_complete(result) and result["source_replay_diagnostic_passed"]
            and all(values[stream_numeric_key] is True and values["stored_update_target_met"] is True
                    and values["fourfold_target_met"] is True
                    and values["selector_diagnostics"]["rank_shortfall_fallback_used"] is False
                    and result["row_preflight"][stream]["fit_conditioning_target_met"] is True
                    for stream, values in result["streams"].items()))


def complete_source_selection(selection, keep, inner):
    """Preserve successful Gram pivots; explicitly fill rank-shortfall in source order.

    No channel synthesis, keep reduction, held-out input or numerical success claim.
    """
    if (selection.dtype != torch.int64 or selection.ndim != 1 or selection.numel() > keep
            or selection.unique().numel() != selection.numel()
            or bool((selection < 0).any()) or bool((selection >= inner).any())):
        raise ValueError("invalid source-channel selector output")
    count = selection.numel()
    if count < keep:
        used = set(selection.tolist())
        fill = torch.tensor([i for i in range(inner) if i not in used][:keep-count], dtype=torch.int64)
        selection = torch.cat((selection, fill))
    return selection.sort().values, {"rank_shortfall_fallback_used": count < keep,
        "pivot_channels": count, "fallback": "retain existing pivots then fill unchanged source channels in source order"}
MIN_FIT_ROWS_PER_KEEP = 2
MAX_ROWS_PER_STREAM = 131072
CAPTURE_KEYS = {"source_ids", "rotation_generated", "rotation_used", "temb", "text_attention_mask",
                "prediction", "sigma", "model_input", "image_patch_indices", "joint_row_kind",
                "context_token_ids", "context_valid"} | {stream + suffix for stream in STREAMS for suffix in
                ("_in", "_out", "_attn_out", "_mlp_in", "_mlp_out", "_modulation")}


class FoldRejected(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def json_file(path, *, expected_sha256=None):
    if path.stat().st_size > 16 << 20:
        raise ValueError("receipt/policy exceeds 16 MiB")
    def bad_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    raw = path.read_bytes()
    if len(raw) > 16 << 20 or (expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError("JSON identity/size mismatch")
    return json.loads(raw.decode("utf-8"), parse_constant=bad_constant)


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("valid lowercase SHA256 required")
    return value


def artifact(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("capture artifact must be a contained filename")
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise ValueError("capture artifact escapes its receipt directory or is missing")
    return path


def load_policy(path, expected_sha256):
    policy = json_file(path, expected_sha256=digest(expected_sha256))
    if set(policy) - {"schema", "preregistered_before_scoring", "gate_scope", "streams", "policy_id", "description"}:
        raise ValueError("unsupported top-level policy fields")
    if policy.get("schema") != POLICY_SCHEMA or policy.get("preregistered_before_scoring") is not True:
        raise ValueError("an explicitly preregistered caller policy is required")
    if policy.get("gate_scope") != "pooled_and_each_heldout_forward":
        raise ValueError("policy must gate pooled and every held-out forward")
    if set(policy.get("streams", {})) != set(STREAMS):
        raise ValueError("policy must declare both image and text streams")
    for stream in STREAMS:
        entry = policy["streams"][stream]
        if set(entry) != {"keep", "max_source_replay_rel_l2", "max_heldout_rel_l2", "min_heldout_cos"}:
            raise ValueError("unsupported/missing stream policy fields; no criterion may be silently ignored")
        keep = entry.get("keep")
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("each stream requires an explicit positive integer keep")
        for key in ("max_source_replay_rel_l2", "max_heldout_rel_l2", "min_heldout_cos"):
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("all numerical gates must be explicit finite caller values")
            if (key.startswith("max_") and value < 0) or (key == "min_heldout_cos" and not -1 <= value <= 1):
                raise ValueError("invalid caller numerical gate")
    return policy


def frozen_shapes(block, width, inner, head_dim):
    prefix = f"transformer_blocks.{block}."
    result = {}
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out.0", "to_add_out"):
        result[prefix + f"attn.{name}.weight"] = (width, width)
        result[prefix + f"attn.{name}.bias"] = (width,)
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        result[prefix + f"attn.{name}.weight"] = (head_dim,)
    for stream in STREAMS:
        result.update({prefix + stream + "_mlp.net.0.proj.weight": (inner, width),
                       prefix + stream + "_mlp.net.0.proj.bias": (inner,),
                       prefix + stream + "_mlp.net.2.weight": (width, inner),
                       prefix + stream + "_mlp.net.2.bias": (width,),
                       prefix + stream + "_mod.1.weight": (6 * width, width),
                       prefix + stream + "_mod.1.bias": (6 * width,)})
    for index, inputs in ((1, 256), (2, width)):
        result[f"time_text_embed.timestep_embedder.linear_{index}.weight"] = (width, inputs)
        result[f"time_text_embed.timestep_embedder.linear_{index}.bias"] = (width,)
    return result


def verified_tensors(root, record, max_bytes):
    path = artifact(root, record["file"])
    if path.stat().st_size > max_bytes or file_sha(path) != digest(record["file_sha256"]):
        raise ValueError("capture artifact size/hash mismatch")
    state = load_file(str(path), device="cpu")
    expected = record.get("tensor_sha256", {})
    if set(state) != set(expected) or set(state) != set(record.get("shapes", {})) or set(state) != set(record.get("dtypes", {})):
        raise ValueError("capture requires complete tensor identities, shapes and dtypes")
    for name, value in state.items():
        if tensor_sha(value) != digest(expected[name]) or list(value.shape) != record["shapes"][name] or str(value.dtype) != record["dtypes"][name]:
            raise ValueError(f"captured tensor identity mismatch: {name}")
    return state, path


def load_captures(paths, *, fixture=False):
    """All-fit/identity conflicts refuse before loading large tensor payloads."""
    if not paths or len(paths) > 32:
        raise ValueError("one to 32 capture receipts required")
    receipts, receipt_identities, pending, image_splits, seen_files = [], [], [], {}, set()
    source_identity = None
    for path in paths:
        receipt_digest = file_sha(path)
        body = json_file(path, expected_sha256=receipt_digest)
        receipt_identities.append({"path": str(path.resolve()), "sha256": receipt_digest})
        if body.get("schema") != CAPTURE_SCHEMA or body.get("status") != "complete" or body.get("fixture") is not fixture:
            raise ValueError("complete authentic capture receipts required; fixtures are not production evidence")
        runtime = body.get("runtime", {})
        if runtime.get("manual_cast_dtype") != "torch.bfloat16" or runtime.get("fp8_optimizations") is not False or runtime.get("selected_block_and_time_match_source_bytes") is not True:
            raise ValueError("captured BF16 manual-cast FP8 source identity is required")
        if not fixture and (body.get("source_code_sha256") != SOURCE_FILES or
                            body["source"]["source_identity"].get("header_sha256") != PARENT_HEADER_SHA256):
            raise ValueError("capture is not the source-pinned QIMG parent")
        source = body["source"]
        if not fixture and (runtime.get("full_residency_verified") is not True or runtime.get("dynamic_patcher") is not False
                or runtime.get("weight_dtype_requested") != "torch.float8_e4m3fn"
                or runtime.get("parent_class") != "comfy.model_base.QwenImage"
                or body.get("source_header_stable_after_capture") is not True
                or isinstance(body["block"], bool) or not isinstance(body["block"], int)
                or not 0 <= body["block"] < 60 or source.get("depth") != 60):
            raise ValueError("complete pinned source inference/residency contract required")
        identity = (body["block"], source["width"], source["head_dim"], source["mlps"]["image"]["parent_inner"],
                    source["mlps"]["text"]["parent_inner"], digest(body["frozen_source"]["file_sha256"]))
        if source_identity is not None and identity != source_identity:
            raise ValueError("capture receipts disagree on frozen source block/time or geometry")
        source_identity = identity
        if identity[3] != identity[4] or (not fixture and identity[1:5] != (3072, 128, 12288, 12288)):
            raise ValueError("unsupported QIMG source geometry")
        upstream = {e["id"]: e for e in body["inputs"]["entries"]}
        if len(upstream) != len(body["inputs"]["entries"]):
            raise ValueError("duplicate input entry identities")
        for record in body.get("captures", []):
            image, prompt, file_digest = (digest(record[k]) for k in ("image_sha256", "prompt_sha256", "file_sha256"))
            split = record.get("split")
            linked = upstream.get(record["sample_id"], {})
            if split not in {"fit", "holdout"} or linked.get("split") != split or linked.get("image", {}).get("sha256") != image or linked.get("prompt_sha256") != prompt:
                raise ValueError("capture image/prompt/split does not match its input provenance")
            if image in image_splits and image_splits[image] != split:
                raise ValueError("the same image identity leaks across fit/holdout")
            image_splits[image] = split
            if file_digest in seen_files:
                raise ValueError("duplicate captured forward cannot be counted twice")
            seen_files.add(file_digest)
            if record.get("fixture") is not fixture or record.get("complete_context") is not True or record.get("row_order") != "text_then_image":
                raise ValueError("complete source joint-context capture required")
            pending.append((path.parent, record))
        receipts.append((path, body))
    if set(image_splits.values()) != {"fit", "holdout"}:
        raise ValueError("image-disjoint fit AND held-out images are mandatory; the all-fit canary cannot fold")
    if not 1 <= len(pending) <= 1024:
        raise ValueError("bounded captured forwards required")
    block, width, head_dim, inner, _, frozen_digest = source_identity
    frozen = None
    for path, body in receipts:
        state, _ = verified_tensors(path.parent, body["frozen_source"], 2 << 30)
        if {k: tuple(v.shape) for k, v in state.items()} != frozen_shapes(block, width, inner, head_dim):
            raise ValueError("frozen block/time does not have the complete dual-stream source geometry")
        if any(v.dtype != torch.float8_e4m3fn for v in state.values()):
            raise ValueError("frozen source tensors must retain original FP8 storage")
        for value in state.values():
            for chunk in value.flatten().split(1 << 20):
                if not bool(torch.isfinite(chunk.float()).all()):
                    raise ValueError("non-finite frozen source tensor")
        if frozen is None:
            frozen = state
        del state
    forwards, row_counts = [], {s: {"fit": 0, "holdout": 0} for s in STREAMS}
    for root, record in pending:
        state, path = verified_tensors(root, record, 256 << 20)
        if set(state) != CAPTURE_KEYS:
            raise ValueError("capture lacks complete original joint-context tensors")
        item = {"file": str(path), "file_sha256": record["file_sha256"], "image_sha256": record["image_sha256"],
                "sample_id": record["sample_id"], "split": record["split"], "sigma": record["sigma"]}
        sigma = state["sigma"]
        if sigma.shape != (1,) or sigma.dtype != torch.float32 or not bool(torch.isfinite(sigma).all()) or not 0 <= float(sigma[0]) <= 1 or float(sigma[0]) != record["sigma"]:
            raise ValueError("captured sigma identity/shape mismatch")
        valid = state["context_valid"]
        if valid.dtype != torch.bool or valid.shape != (1, record["text_tokens"]) or not bool(valid.any()):
            raise ValueError("actual text validity mask required")
        total = record["image_tokens"] + record["text_tokens"]
        if state["source_ids"].shape != (1, total, 3) or state["rotation_used"].shape != (1, 1, total, head_dim // 2, 2, 2):
            raise ValueError("source coordinates/rotation do not span the complete joint context")
        if not torch.equal(state["rotation_generated"].to(state["rotation_used"].dtype), state["rotation_used"]):
            raise ValueError("source-generated and source-used rotations disagree")
        for stream, tokens in (("img", record["image_tokens"]), ("txt", record["text_tokens"])):
            x, y = state[stream + "_mlp_in"], state[stream + "_mlp_out"]
            if x.shape != (1, tokens, width) or y.shape != x.shape or x.dtype != torch.bfloat16 or y.dtype != x.dtype:
                raise ValueError("complete BF16 MLP source rows required")
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
                raise ValueError("non-finite source MLP rows")
            mask = valid.flatten() if stream == "txt" else torch.ones(tokens, dtype=torch.bool)
            item[stream] = {"x": x, "y": y, "valid": mask}
            row_counts[stream][record["split"]] += int(mask.sum())
            if sum(row_counts[stream].values()) > MAX_ROWS_PER_STREAM:
                raise ValueError("too many retained rows for bounded folding memory")
        forwards.append(item)
    return {"frozen": frozen, "frozen_file_sha256": frozen_digest, "block": block, "width": width, "inner": inner,
            "forwards": forwards, "row_counts": row_counts, "image_splits": image_splits, "fixture": fixture,
            "capture_receipts": receipt_identities}


def row_preflight(data, policy):
    report = {}
    for stream in STREAMS:
        keep = policy["streams"][stream]["keep"]
        rows = data["row_counts"][stream]
        maximum = maximum_fourfold_keep(data["width"], data["inner"])
        if isinstance(keep, bool) or not isinstance(keep, int) or not 1 <= keep <= data["inner"]:
            raise ValueError(f"{stream}: keep must be within the source channel geometry")
        required = MIN_FIT_ROWS_PER_KEEP * keep
        if rows["fit"] < 1 or rows["holdout"] < 1:
            raise ValueError(f"{stream}: nonempty genuine FIT and image-disjoint holdout rows required")
        report[stream] = {**rows, "keep": keep, "minimum_fit_rows": required,
                          "fit_conditioning_target_met": rows["fit"] >= required,
                          "fourfold_target_met": keep <= maximum,
                          "targets_are_diagnostics": True,
                          "maximum_keep_by_count": maximum, "maximum_keep_by_available_fit_rows": rows["fit"] // 2}
    return report


def mlp_weights(frozen, block, stream, device):
    prefix = f"transformer_blocks.{block}.{stream}_mlp."
    return tuple(frozen[prefix + name].to(device=device, dtype=torch.bfloat16) for name in
                 ("net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"))


def mlp_forward(x, weights):
    """Pinned Comfy manual-cast Linear -> GELU(tanh) -> Linear boundaries."""
    w1, b1, w2, b2 = weights
    value = x.to(device=w1.device, dtype=torch.bfloat16)
    hidden = F.gelu(F.linear(value, w1, b1), approximate="tanh")
    output = F.linear(hidden, w2, b2)
    if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(output).all()):
        raise ValueError("non-finite source/candidate MLP evaluation")
    return hidden, output


def score(actual, target):
    if actual.shape != target.shape:
        raise ValueError("evaluation tensor shape mismatch")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("non-finite actual or target evaluation tensor")
    result = write_scores(actual.detach().cpu().reshape(-1, actual.shape[-1]),
                          target.detach().cpu().reshape(-1, target.shape[-1]))
    return json_diagnostics(result)


def numerical_pass(scores, policy):
    cosine = scores["cos"]
    return (finite_at_most(scores["rel_l2"], policy["max_heldout_rel_l2"])
            and isinstance(cosine, (int, float)) and not isinstance(cosine, bool)
            and math.isfinite(cosine) and cosine >= policy["min_heldout_cos"])


@torch.no_grad()
def fold_loaded(data, policy, *, device="cpu"):
    rows = row_preflight(data, policy)
    baseline, features = {}, {}
    # BOTH streams must replay before EITHER stream reaches pivoting/refitting.
    for stream in STREAMS:
        weights = mlp_weights(data["frozen"], data["block"], stream, device)
        baseline[stream], fit_hidden = [], []
        for item in data["forwards"]:
            captured = item[stream]
            hidden, predicted = mlp_forward(captured["x"], weights)
            scores = score(predicted, captured["y"])
            baseline[stream].append({"file": item["file"], "split": item["split"], "scores": scores})
            baseline[stream][-1]["diagnostic_target_met"] = finite_at_most(scores["rel_l2"], policy["streams"][stream]["max_source_replay_rel_l2"])
            if item["split"] == "fit":
                fit_hidden.append(hidden.squeeze(0).cpu()[captured["valid"]])
        features[stream] = torch.cat(fit_hidden)
        del weights, fit_hidden
    candidate = dict(data["frozen"])
    selections, measured = {}, {}
    for stream in STREAMS:
        entry = policy["streams"][stream]
        keep, width, inner = entry["keep"], data["width"], data["inner"]
        hidden = features.pop(stream)
        gram = torch.zeros((inner, inner), dtype=torch.float64)
        for values in hidden.split(512):
            values = values.double()
            gram.add_(values.T @ values)
        selection = pivoted_columns_gram(gram, keep)
        selection, selector_diagnostics = complete_source_selection(selection, keep, inner)
        del gram
        prefix = f"transformer_blocks.{data['block']}.{stream}_mlp."
        # Bias is a counted coefficient and has its own prior column. Neither
        # the target bias nor the parent anchor is silently discarded/zeroed.
        prior_w = data["frozen"][prefix + "net.2.weight"].to(torch.bfloat16).double()[:, selection]
        prior_b = data["frozen"][prefix + "net.2.bias"].to(torch.bfloat16).double()[:, None]
        prior = torch.cat((prior_w, prior_b), dim=1)
        design = torch.cat((hidden[:, selection].double(), torch.ones((len(hidden), 1), dtype=torch.float64)), dim=1)
        targets = torch.cat([item[stream]["y"].squeeze(0)[item[stream]["valid"]]
                             for item in data["forwards"] if item["split"] == "fit"]).double()
        fitted, solve = law_abiding_solve(design, targets, prior)
        solve = json_diagnostics(solve)
        if not bool(torch.isfinite(fitted).all()):
            raise ValueError("non-finite closed-form solution")
        stored = fitted.to(torch.bfloat16)
        stored_update = float((stored.double() - prior).norm() / prior.norm().clamp_min(1e-30))
        if not bool(torch.isfinite(stored).all()):
            raise ValueError("closed-form coefficients overflowed BF16 storage")
        for name in ("net.0.proj.weight", "net.0.proj.bias"):
            original = data["frozen"][prefix + name]
            # CPU advanced indexing lacks FP8 support in some pinned runtimes;
            # gather the original one-byte representation without quantization.
            candidate[prefix + name] = original.view(torch.uint8).index_select(0, selection).contiguous().view(original.dtype)
        candidate[prefix + "net.2.weight"] = stored[:, :-1].contiguous()
        candidate[prefix + "net.2.bias"] = stored[:, -1].contiguous()
        del fitted, design, targets, hidden, prior_w, prior_b, prior
        weights = mlp_weights(candidate, data["block"], stream, device)
        per_forward, pooled = [], {split: [[], []] for split in ("fit", "holdout")}
        for item in data["forwards"]:
            captured = item[stream]
            _, prediction = mlp_forward(captured["x"], weights)
            prediction = prediction.squeeze(0).cpu()[captured["valid"]]
            target = captured["y"].squeeze(0)[captured["valid"]]
            scores = score(prediction, target)
            per_forward.append({"file": item["file"], "image_sha256": item["image_sha256"],
                                "split": item["split"], "sigma": item["sigma"], "scores": scores})
            pooled[item["split"]][0].append(prediction)
            pooled[item["split"]][1].append(target)
        fit_scores, held_scores = (score(torch.cat(pooled[split][0]), torch.cat(pooled[split][1])) for split in ("fit", "holdout"))
        passed = numerical_pass(held_scores, entry) and all(numerical_pass(row["scores"], entry) for row in per_forward if row["split"] == "holdout")
        parent_count = biased_gelu_mlp_parameters(width, inner)
        child_count = sum(candidate[prefix + name].numel() for name in
                          ("net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"))
        compression_target_met = 4 * child_count <= parent_count
        measured[stream] = {"keep": keep, "parent_parameters": parent_count, "child_parameters": child_count,
            "component_compression_x": parent_count / child_count, "solve": solve,
            "selector_diagnostics": selector_diagnostics, "fourfold_target_met": compression_target_met,
            "stored_update_scale": stored_update, "stored_update_target_met": finite_at_most(stored_update, 1),
            "fit_scores": fit_scores, "heldout_scores": held_scores,
            "per_forward": per_forward, "numerical_policy_passed": passed}
        selections[stream + ".selected_channels"] = selection
        del weights, pooled
    result = {"schema": OWNER_FOLD_SCHEMA, "status": "measured_candidate", "fixture": data["fixture"],
        **current_artifact_fields(), "artifact_admitted": False, "serialization_validation": "not yet written",
        "block": data["block"], "source_replay": baseline, "streams": measured, "row_preflight": rows,
        "policy": policy, "image_splits": data["image_splits"], "capture_receipts": data["capture_receipts"],
        "source_frozen_file_sha256": data["frozen_file_sha256"],
        "source_parent_identity_scope": "capture parent header plus complete selected block/time and captured-file hashes; not a newly verified full parent-file digest",
        "numerical_policy_passed": all(m["numerical_policy_passed"] for m in measured.values()),
        "runtime": {"device": device, "torch": torch.__version__, "compute_dtype": "torch.bfloat16",
                    "solver_dtype": "torch.float64", "torch_threads": torch.get_num_threads()},
        "law": {"selection": "existing pivoted_columns_gram on FIT-only BF16 source GELU features",
                "refit": "existing law_abiding_solve, with parent-anchored affine bias column",
                "min_fit_rows_per_keep": MIN_FIT_ROWS_PER_KEEP, "heldout_used_for_selection_or_refit": False,
                "text_rows": "valid context rows only for fit/heldout; source replay covers all captured text rows"},
        "accounting": {"parent_block_and_time_parameters": sum(t.numel() for t in data["frozen"].values()),
                       "child_block_and_time_parameters": sum(t.numel() for t in candidate.values()),
                       "selection_indices_are_parameters": False},
        "runtime_requirements": {"native_qimg_dual_stream_runtime": "still required; generic adaln6 is insufficient",
                                 "paired_host_source_charts": "not supplied or invented", "full_joint_block_replay": "not performed by this token-local MLP fold",
                                 "weight_storage": "preserved FP8 source plus BF16 refit; a loader that globally recasts this candidate to FP8 is not equivalent"},
        "capability_claim": "none: measured MLP candidate is not installed, behaviorally accepted, or a media/all-source child proof"}
    result["source_replay_diagnostic_passed"] = all(row["diagnostic_target_met"] for stream in baseline.values() for row in stream)
    result["diagnostic_measurements_complete"] = diagnostics_complete(result)
    result["all_numeric_diagnostic_targets_met"] = all_numeric_targets_met(result, "numerical_policy_passed")
    return candidate, selections, json_diagnostics(result)


def implementation_sha256():
    return {"c16_qimg_fold.py": file_sha(Path(__file__)),
            "c16_qimg_capture.py": file_sha(TOOLS / "c16_qimg_capture.py"),
            "mediafold/linalg.py": file_sha(TOOLS.parents[1] / "media/mediafold/linalg.py")}


def destinations_preflight(receipts, policy_path, out, rejection_receipt):
    """Refuse aliases/overwrites before reading large captures or fitting."""
    if out.exists() or out.is_symlink():
        raise ValueError("output must be a new directory; no overwrite/cleanup")
    rejection_receipt = rejection_receipt if rejection_receipt is not None else out.with_name(out.name + ".rejected.json")
    if rejection_receipt.exists() or rejection_receipt.is_symlink():
        raise ValueError("rejection receipt must be a new file; no overwrite/cleanup")
    output, rejection = out.resolve(), rejection_receipt.resolve()
    def overlaps(left, right):
        return left == right or left in right.parents or right in left.parents
    if overlaps(output, rejection):
        raise ValueError("rejection receipt must be separate from the candidate output directory")
    # Frozen source and forward tensors must be contained in receipt directories.
    # Do not create a report or any of its parents inside those source directories.
    source_directories = [*(path.parent.resolve() for path in receipts), TOOLS.resolve(),
                          (TOOLS.parents[1] / "media").resolve()]
    input_files = [*(path.resolve() for path in receipts), policy_path.resolve()]
    if any(overlaps(destination, source) for destination in (output, rejection)
           for source in source_directories + input_files):
        raise ValueError("output/rejection receipt overlaps an input or source directory")
    return rejection


def convert(receipts, policy_path, policy_sha256, out, *, device="cpu", validate_only=False, fixture=False,
            rejection_receipt=None):
    rejection_receipt = destinations_preflight(receipts, policy_path, out, rejection_receipt)
    policy = load_policy(policy_path, policy_sha256)
    data = load_captures(receipts, fixture=fixture)
    preflight = row_preflight(data, policy)
    if validate_only:
        return {"validated": True, "fold_launched": False, "fixture": fixture, "row_preflight": preflight,
                "source_replay_performed": False, "policy_sha256": policy_sha256, "capture_receipts": data["capture_receipts"]}
    try:
        candidate, selection, result = fold_loaded(data, policy, device=device)
    except FoldRejected as exc:
        exc.receipt.update({"schema": FOLD_SCHEMA, "fixture": fixture, "policy": policy,
            "policy_sha256": policy_sha256, "capture_receipts": data["capture_receipts"],
            "source_frozen_file_sha256": data["frozen_file_sha256"], "candidate_written": False,
            "rejection_reason": str(exc), "rejection_receipt": str(rejection_receipt),
            "implementation_sha256": implementation_sha256()})
        # Persistence precedes stdout, so detached/lost logs cannot erase the
        # measured refusal. Serialize before creating parents; never overwrite.
        payload = json.dumps(exc.receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
        destinations_preflight(receipts, policy_path, out, rejection_receipt)
        rejection_receipt.parent.mkdir(parents=True, exist_ok=True)
        with rejection_receipt.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        raise
    destinations_preflight(receipts, policy_path, out, rejection_receipt)
    out.mkdir(parents=True, exist_ok=False)
    result["policy_sha256"] = policy_sha256
    result["policy_preregistration_scope"] = "caller-pinned bytes; chronological preregistration is an external review obligation"
    result["implementation_sha256"] = implementation_sha256()
    try:
        result["output"] = write_tensors(out / "folded_source_block_and_time.safetensors", candidate)
        result["selection"] = write_tensors(out / "selection.safetensors", selection)
        for key, expected in (("output", candidate), ("selection", selection)):
            stored, _ = verified_tensors(out, result[key], 2 << 30)
            if set(stored) != set(expected) or any(tensor_sha(stored[k]) != tensor_sha(v) for k, v in expected.items()):
                raise ValueError("serialized candidate or selection changed on readback")
            del stored
        result.update(candidate_written=True, selection_tensors_written=True,
                      serialization_validation="complete: exact tensor identities/shapes/dtypes and byte readback",
                      artifact_admitted=not fixture, structural_artifact_valid=True)
    except Exception as exc:
        result["status"] = "incomplete_output"
        result["structural_artifact_valid"] = False
        result["artifact_admitted"] = False
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (out / "fold_receipt.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", nargs="+", type=Path, required=True, help="completed capture_receipt.json files")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True, help="externally preregistered policy digest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rejection-receipt", type=Path,
                        help="exclusive refusal JSON, default sibling <out>.rejected.json; never a candidate")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        result = convert(args.captures, args.policy, args.policy_sha256, args.out,
                         device=args.device, validate_only=args.validate_only,
                         rejection_receipt=args.rejection_receipt)
    except FoldRejected as exc:
        print(json.dumps(exc.receipt, sort_keys=True, indent=2, allow_nan=False))
        raise SystemExit(2) from exc
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
