#!/usr/bin/env python3
"""Replay a complete pinned native QIMG block; report, never accept, its fold.

No full parent, sampler, training, token subsampling, generic adaln6 conversion,
GPU allocator, or installer is involved. A caller supplies the baseline tolerance.
The original block must replay before the folded block is evaluated at all.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
from pathlib import Path
import sys

import torch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import c16_qimg_fold as fold

SCHEMA = "neural_forge.qimg_native_joint_replay.v1"
SOURCE_FILES = {**fold.SOURCE_FILES,
    "comfy/ops.py": "aa6dc3dd7f2d6b04f9e3f5df949348a4734f5f9af4d08891c57c8a64986d3782",
    "comfy/ldm/common_dit.py": "5e2459be4a47073a17e6b806f2b93962b4df1cdcaa2ac1890c9573bf942edf82",
    "comfy/ldm/modules/attention.py": "a540a0b487eb5979c4dbc38c72926e3a38b5e41480a038a98707e2eef7e038d9"}
TAPS = tuple(s + suffix for s in fold.STREAMS for suffix in
             ("_out", "_attn_out", "_mlp_in", "_mlp_out", "_modulation"))


def same_bytes(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.detach().cpu().contiguous().view(torch.uint8), right.detach().cpu().contiguous().view(torch.uint8))


def load_candidate(path, expected_sha256, data):
    body = fold.json_file(path, expected_sha256=fold.digest(expected_sha256))
    if (body.get("schema") != fold.FOLD_SCHEMA or body.get("status") != "measured_candidate"
            or body.get("fixture") is not data["fixture"] or body.get("numerical_policy_passed") is not True):
        raise ValueError("an exact pinned measured-candidate fold receipt is required")
    if body.get("block") != data["block"] or body.get("source_frozen_file_sha256") != data["frozen_file_sha256"]:
        raise ValueError("fold belongs to another source block/time")
    identities = sorted(item["sha256"] for item in data["capture_receipts"])
    if sorted(item["sha256"] for item in body.get("capture_receipts", [])) != identities:
        raise ValueError("fold and replay must reference the exact same capture receipts")
    candidate, _ = fold.verified_tensors(path.parent, body["output"], 2 << 30)
    selected, _ = fold.verified_tensors(path.parent, body["selection"], 1 << 20)
    if set(candidate) != set(data["frozen"]) or set(selected) != {s + ".selected_channels" for s in fold.STREAMS}:
        raise ValueError("incomplete candidate or selections")
    allowed = set()
    for stream in fold.STREAMS:
        prefix = f"transformer_blocks.{data['block']}.{stream}_mlp."
        keep = body["streams"][stream]["keep"]
        if isinstance(keep, bool) or not isinstance(keep, int) or not 1 <= keep <= fold.maximum_fourfold_keep(data["width"], data["inner"]):
            raise ValueError("invalid bias-inclusive folded width")
        index = selected[stream + ".selected_channels"]
        if (index.dtype != torch.int64 or index.shape != (keep,) or index.unique().numel() != keep
                or bool((index < 0).any()) or bool((index >= data["inner"]).any())
                or not torch.equal(index, index.sort().values)):
            raise ValueError("invalid selected source channels")
        for suffix in ("net.0.proj.weight", "net.0.proj.bias"):
            key = prefix + suffix
            source = data["frozen"][key]
            wanted = source.view(torch.uint8).index_select(0, index).contiguous().view(source.dtype)
            if not same_bytes(candidate[key], wanted):
                raise ValueError("first-layer selected source bytes changed")
            allowed.add(key)
        for suffix, shape in (("net.2.weight", (data["width"], keep)), ("net.2.bias", (data["width"],))):
            key = prefix + suffix
            if candidate[key].dtype != torch.bfloat16 or candidate[key].shape != shape or not bool(torch.isfinite(candidate[key]).all()):
                raise ValueError("refit must retain finite BF16 weight and bias with its actual width")
            allowed.add(key)
        prior = torch.cat((data["frozen"][prefix + "net.2.weight"].bfloat16().double()[:, index],
                           data["frozen"][prefix + "net.2.bias"].bfloat16().double()[:, None]), dim=1)
        actual = torch.cat((candidate[prefix + "net.2.weight"].double(), candidate[prefix + "net.2.bias"].double()[:, None]), dim=1)
        if float((actual - prior).norm() / prior.norm().clamp_min(1e-30)) > 1:
            raise ValueError("stored candidate violates the existing update veto")
    for key, value in data["frozen"].items():
        if key not in allowed and not same_bytes(candidate[key], value):
            raise ValueError("candidate changed a frozen attention/norm/modulation/time tensor")
    return candidate, {"path": str(path.resolve()), "sha256": expected_sha256,
                       "output_sha256": body["output"]["file_sha256"], "selection_sha256": body["selection"]["file_sha256"]}


def source_modules_inside(root):
    for name, module in tuple(sys.modules.items()):
        if (name == "comfy" or name.startswith("comfy.")) and getattr(module, "__file__", None):
            if root not in Path(module.__file__).resolve().parents:
                raise ValueError("Comfy was already imported from a different source tree")


def init_native(root, device, capture_runtime):
    root = root.resolve()
    actual = {name: fold.file_sha(root / name) for name in SOURCE_FILES}
    if actual != SOURCE_FILES:
        raise ValueError("native replay source differs from the pinned QIMG/ops/attention contract")
    source_modules_inside(root)
    if torch.__version__ != capture_runtime["torch"]:
        raise ValueError("native replay requires the captured Torch build")
    sys.path.insert(0, str(root))
    import comfy.options
    from comfy.cli_args import args
    args.cpu = device == "cpu"
    try:
        import comfy.model_management as management
    except RuntimeError as exc:
        if device == "cpu" and "0 active drivers" in str(exc):
            raise RuntimeError("this pinned Comfy/Triton image cannot import native modules without a GPU driver; no backend patch or baseline waiver is applied") from exc
        raise
    import comfy.ops as ops
    model = importlib.import_module("comfy.ldm.qwen_image.model")
    source_modules_inside(root)
    if management.in_training or str(management.get_torch_device()) != device:
        raise ValueError("wrong native device or training kernel path")
    torch.set_float32_matmul_precision(capture_runtime["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = capture_runtime["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = capture_runtime["cudnn_allow_tf32"]
    expected = {"hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask", "temb",
                "image_rotary_emb", "timestep_zero_index", "transformer_options"}
    if set(inspect.signature(model.QwenImageTransformerBlock.forward).parameters) != expected | {"self"}:
        raise ValueError("unexpected native block API")
    return lambda state, data: NativeBlock(model, ops.manual_cast, state, data, device), actual


class NativeBlock:
    """Only native classes; assign=True deliberately preserves mixed storage."""
    def __init__(self, model, operations, state, data, device, *, axes=(16, 56, 56)):
        self.device, self.index = device, data["block"]
        prefix = f"transformer_blocks.{self.index}."
        width = data["width"]
        head = state[prefix + "attn.norm_q.weight"].numel()
        if sum(axes) != head or width % head:
            raise ValueError("source head/rotary geometry mismatch")
        kwargs = {"dtype": torch.bfloat16, "device": "meta", "operations": operations}
        self.block = model.QwenImageTransformerBlock(dim=width, num_attention_heads=width // head,
                                                    attention_head_dim=head, **kwargs)
        for stream in fold.STREAMS:
            inner = state[prefix + stream + "_mlp.net.0.proj.weight"].shape[0]
            if inner != 4 * width:
                setattr(self.block, stream + "_mlp", model.FeedForward(dim=width, dim_out=width, inner_dim=inner, **kwargs))
        self.time = model.QwenTimestepProjEmbeddings(embedding_dim=width, pooled_projection_dim=768,
                                                    use_additional_t_cond=False, **kwargs)
        self.embed = model.EmbedND(dim=head, theta=10000, axes_dim=list(axes))
        for module, key_prefix in ((self.block, prefix), (self.time, "time_text_embed.")):
            values = {k[len(key_prefix):]: v.to(device=device) for k, v in state.items() if k.startswith(key_prefix)}
            module.load_state_dict(values, strict=True, assign=True)
            module.eval().requires_grad_(False)
            if set(module.state_dict()) != set(values) or any(not same_bytes(v, values[k]) for k, v in module.state_dict().items()):
                raise ValueError("native loader altered a source/refit dtype or coefficient")

    @torch.no_grad()
    def forward(self, state, record):
        captured = {}
        def put(name, value):
            if name in captured:
                raise ValueError("native hook fired twice")
            captured[name] = value.detach().cpu().clone()
        handles = []
        try:
            for stream in fold.STREAMS:
                mlp, mod = getattr(self.block, stream + "_mlp"), getattr(self.block, stream + "_mod")
                handles.append(mlp.register_forward_pre_hook(lambda m, args, s=stream: put(s + "_mlp_in", args[0])))
                handles.append(mlp.register_forward_hook(lambda m, args, out, s=stream: put(s + "_mlp_out", out)))
                handles.append(mod.register_forward_hook(lambda m, args, out, s=stream: put(s + "_modulation", out)))
            def attention_output(module, args, out):
                put("img_attn_out", out[0]); put("txt_attn_out", out[1])
            handles.append(self.block.attn.register_forward_hook(attention_output))
            def arg(name):
                # Native operations cannot mutate the original captured targets.
                return state[name].to(device=self.device).clone()
            img, txt, temb, pe = (arg(k) for k in ("img_in", "txt_in", "temb", "rotation_used"))
            mask = arg("text_attention_mask") if record["text_mask_present"] else None
            put("time_recomputed", self.time(arg("sigma"), img))
            put("rotation_recomputed", self.embed(arg("source_ids")))
            txt_out, img_out = self.block(hidden_states=img, encoder_hidden_states=txt,
                encoder_hidden_states_mask=mask, temb=temb, image_rotary_emb=pe, timestep_zero_index=None,
                transformer_options={"total_blocks": 60, "block_type": "double", "block_index": self.index})
            put("txt_out", txt_out); put("img_out", img_out)
        finally:
            for handle in handles:
                handle.remove()
        return captured


def context_contract(state, record, data):
    width, ni, nt = data["width"], record["image_tokens"], record["text_tokens"]
    if record.get("text_mask_present") not in (True, False) or type(record.get("text_mask_present")) is not bool:
        raise ValueError("explicit native mask-presence metadata required")
    for stream, count in (("img", ni), ("txt", nt)):
        for suffix in ("_in", "_out", "_attn_out", "_mlp_in", "_mlp_out"):
            tensor = state[stream + suffix]
            if tensor.shape != (1, count, width) or tensor.dtype != torch.bfloat16 or not bool(torch.isfinite(tensor).all()):
                raise ValueError("full finite BF16 stream context is required")
        if state[stream + "_modulation"].shape != (1, width * 6):
            raise ValueError("native source modulation shape mismatch")
    if state["temb"].shape != (1, width) or state["temb"].dtype != torch.bfloat16 or not bool(torch.isfinite(state["temb"]).all()):
        raise ValueError("invalid captured time embedding")
    valid, mask = state["context_valid"], state["text_attention_mask"]
    if valid.dtype != torch.bool or valid.shape != (1, nt) or not bool(valid.any()):
        raise ValueError("invalid actual text validity mask")
    if mask.shape != (1, nt) or mask.dtype != torch.bfloat16 or bool(torch.isnan(mask).any()) or bool(torch.isposinf(mask).any()):
        raise ValueError("invalid native additive mask")
    if record["text_mask_present"]:
        # Finite negative mask biases and -inf are both native additive values;
        # do not turn them into boolean masks or synthesize different biases.
        if not bool(torch.isfinite(mask[valid]).all()) or bool((mask[~valid] > torch.finfo(mask.dtype).min).any()):
            raise ValueError("native mask disagrees with captured validity")
    elif not bool(valid.all()) or not bool((mask == 0).all()):
        raise ValueError("absent native mask must not hide invalid text rows")
    model_input = state["model_input"]
    if (model_input.ndim != 5 or model_input.shape[:3] != (1, 16, 1) or model_input.dtype != torch.bfloat16
            or min(model_input.shape[-2:]) < 2 or any(n % 2 for n in model_input.shape[-2:])
            or not bool(torch.isfinite(model_input).all())):
        raise ValueError("only the complete plain native image grid is supported")
    h, w = model_input.shape[-2] // 2, model_input.shape[-1] // 2
    row, col = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack((torch.zeros(h * w, dtype=torch.int64), row.flatten(), col.flatten()), 1)
    if ni != h * w or not torch.equal(grid, state["image_patch_indices"]):
        raise ValueError("image source grid was sampled, reordered or changed")
    image_ids = grid.float(); image_ids[:, 1] -= h // 2; image_ids[:, 2] -= w // 2
    text_ids = torch.arange(max(h // 2, w // 2), max(h // 2, w // 2) + nt).float()[:, None].expand(-1, 3)
    if state["source_ids"].dtype != torch.float32 or not torch.equal(state["source_ids"], torch.cat((text_ids, image_ids))[None]):
        raise ValueError("captured positions differ from the pinned complete native source grid")
    if not torch.equal(state["joint_row_kind"], torch.tensor([0] * nt + [1] * ni, dtype=torch.int8)):
        raise ValueError("joint context is not text then image")
    head = data["frozen"][f"transformer_blocks.{data['block']}.attn.norm_q.weight"].numel()
    if (state["rotation_used"].dtype != torch.bfloat16 or state["rotation_generated"].dtype != torch.float32
            or state["rotation_used"].shape != (1, 1, ni + nt, head // 2, 2, 2)
            or state["rotation_generated"].shape != state["rotation_used"].shape
            or not torch.equal(state["rotation_generated"].bfloat16(), state["rotation_used"])):
        raise ValueError("source rotation was changed or recast differently")


def capture_jobs(data):
    jobs, runtime = [], None
    for identity in data["capture_receipts"]:
        path = Path(identity["path"])
        body = fold.json_file(path, expected_sha256=identity["sha256"])
        if runtime is not None and body["runtime"] != runtime:
            raise ValueError("capture batches have different native kernel/runtime contracts")
        runtime = body["runtime"]
        jobs.extend((path.parent, record) for record in body["captures"])
    return jobs, runtime


@torch.no_grad()
def evaluate(data, candidate, factory, jobs, max_source_rel_l2):
    if isinstance(max_source_rel_l2, bool) or not isinstance(max_source_rel_l2, (int, float)) or not math.isfinite(max_source_rel_l2) or max_source_rel_l2 < 0:
        raise ValueError("caller must supply an explicit finite nonnegative baseline tolerance")
    baseline = []
    native = factory(data["frozen"], data)
    for root, record in jobs:
        state, _ = fold.verified_tensors(root, record, 256 << 20)
        context_contract(state, record, data)
        actual = native.forward(state, record)
        wanted = {**{k: state[k] for k in TAPS}, "time_recomputed": state["temb"],
                  "rotation_recomputed": state["rotation_generated"]}
        if set(actual) != set(wanted):
            raise ValueError("native replay did not produce every complete source tap")
        metrics = {k: fold.score(actual[k], v) for k, v in wanted.items()}
        baseline.append({"file": record["file"], "file_sha256": record["file_sha256"], "split": record["split"], "scores": metrics})
        if any(value["rel_l2"] > max_source_rel_l2 for value in metrics.values()):
            raise fold.FoldRejected("native original-block baseline failed; no folded block evaluated",
                {"schema": SCHEMA, "status": "rejected_source_replay", "baseline": baseline,
                 "max_source_rel_l2": max_source_rel_l2, "folded_block_evaluated": False})
    del native, actual, state
    native = factory(candidate, data)
    measured, pools = [], {s: {p: [[], []] for p in ("fit", "holdout")} for s in fold.STREAMS}
    for root, record in jobs:
        state, _ = fold.verified_tensors(root, record, 256 << 20)
        context_contract(state, record, data)
        actual = native.forward(state, record)
        metrics = {}
        for stream in fold.STREAMS:
            output, target = actual[stream + "_out"], state[stream + "_out"]
            valid = state["context_valid"].flatten() if stream == "txt" else torch.ones(record["image_tokens"], dtype=torch.bool)
            predicted, wanted = output.squeeze(0)[valid], target.squeeze(0)[valid]
            metrics[stream] = {"all_context_rows": fold.score(output, target), "valid_rows": fold.score(predicted, wanted)}
            pools[stream][record["split"]][0].append(predicted)
            pools[stream][record["split"]][1].append(wanted)
        measured.append({"file": record["file"], "file_sha256": record["file_sha256"],
            "image_sha256": record["image_sha256"], "split": record["split"], "sigma": record["sigma"], "scores": metrics})
    pooled = {s: {p: fold.score(torch.cat(pair[0]), torch.cat(pair[1])) for p, pair in split.items()} for s, split in pools.items()}
    return {"schema": SCHEMA, "status": "measured_native_joint_block", "fixture": data["fixture"],
        "max_source_rel_l2": max_source_rel_l2, "source_baseline_passed": True, "baseline": baseline,
        "folded_per_forward": measured, "folded_pooled_valid_rows": pooled,
        "folded_numerical_acceptance": "not evaluated; no full-block acceptance criteria invented",
        "scope": "complete single source block on captured parent block inputs; not propagation through 60 folded blocks",
        "time_and_positions": "native source time and position embedding rechecked; block receives unchanged captured temb/rotation/mask",
        "capability_claim": "none; no sampler, VAE, paired charts, native child installation, or media quality proof"}


def replay(receipts, folded_receipt, folded_sha256, comfy_root, device, max_source_rel_l2, out, *, fixture=False, factory=None):
    if out.exists() or out.is_symlink():
        raise ValueError("report already exists; no overwrite/cleanup")
    data = fold.load_captures(receipts, fixture=fixture)
    candidate, identity = load_candidate(folded_receipt, folded_sha256, data)
    jobs, runtime = capture_jobs(data)
    # Retain complete capture references, not a second copy of every MLP array.
    del data["forwards"]
    if factory is not None and not fixture:
        raise ValueError("production replay cannot substitute a fixture/native factory")
    source = None
    if factory is None:
        factory, source = init_native(comfy_root, device, runtime)
    try:
        result = evaluate(data, candidate, factory, jobs, max_source_rel_l2)
    except fold.FoldRejected as exc:
        exc.receipt.update({"fixture": fixture, "capture_receipts": data["capture_receipts"], "fold_receipt": identity})
        raise
    result.update({"capture_receipts": data["capture_receipts"], "fold_receipt": identity,
        "source_frozen_file_sha256": data["frozen_file_sha256"], "source_code_sha256": source,
        "row_counts": data["row_counts"], "image_splits": data["image_splits"],
        "runtime": {"device": device, "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "storage": "original FP8 and refit BF16 assigned without global recasting; native manual-cast operations",
            "image_identity_scope": "immutable runtime image identity remains an external launcher obligation"},
        "implementation_sha256": {"c16_qimg_joint_replay.py": fold.file_sha(Path(__file__)),
                                  "c16_qimg_fold.py": fold.file_sha(TOOLS / "c16_qimg_fold.py")}})
    # The report is the only output. Never rewrite weights or previous receipts.
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--fold-receipt", type=Path, required=True)
    parser.add_argument("--fold-receipt-sha256", required=True)
    parser.add_argument("--comfy-root", type=Path, default=Path("/opt/ComfyUI"))
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--max-source-rel-l2", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new JSON report file in an existing directory")
    args = parser.parse_args()
    try:
        result = replay(args.captures, args.fold_receipt, args.fold_receipt_sha256, args.comfy_root,
                        args.device, args.max_source_rel_l2, args.out)
    except fold.FoldRejected as exc:
        print(json.dumps(exc.receipt, sort_keys=True, indent=2, allow_nan=False))
        raise SystemExit(2) from exc
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
