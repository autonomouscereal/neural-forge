"""Source-pinned grouping diagnostic; never an admission gate or production patch.

The authentic replay API accepts one immutable residual pair and loaded parent
runner. It compares whole-corpus and row-group calls to the ORIGINAL engine,
without fitting, modifying parent weights, or relaxing parity thresholds.
The CLI only prepares a plan or runs a small authored CPU arithmetic control.
It does not allocate a GPU or load the real Kimi checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PINS = {
    "k3_dense_pass.py": "fd7dbf8da04504696ebb0702372802d2869bd1c13ee8c50d2578cee8108a55b7",
    "k3_common.py": "d2ef39c46ba9a7b510f2a02fbaaa6b94ab0b339f6ccdf1a66b086c18ba721c33",
    "k3_layer.py": "07269c9196ab6e45c3299d671f972e6d2da296157fe5194e4cee5fd7a61c5390",
    "k3_stream.py": "789cb7fe18cbfa0a5b83b3addd5fc8edd5ccf094edbc5f55766bd4e5dc29b5bf",
    "k3_review_stream.py": "94c2005b80d5ba963cbf3f73ae76be57b1381d74ae984f873a0d48719d35af76",
    "k3_receipt.py": "9242d42cc77a398265a319161e0d8e2fa06a6e94aedd87c2ea6fe19abba7dcd2",
    "c14_union_coreset.py": "3356780e918af111a108d40ef6b2b2658667441256014a9eb3443fa1e78cd535",
    "k3_teacher_cache-gpu6.py": "7899023cb8eb411c4b0fef005bb135970aaef65bb42d4653c688800bf7e88826",
}
FOLD_SHA = "8948bab7feeaa3cd460418f3552600b2d93a6c944143cfedb1b3a9122fa787df"
PROBE_SHA = "a20d5d9a2b88ff269513286f1951a121545143783953aa339aa4dbb225ef3039"


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_sources(directory: Path) -> dict[str, str]:
    observed = {}
    for name, expected in PINS.items():
        path = directory / name
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError("source pin mismatch: " + name)
        observed[name] = expected
    return observed


def make_groups(segments: list[dict], group_tokens: int) -> list[dict]:
    if type(group_tokens) is not int or group_tokens < 1:
        raise ValueError("group_tokens must be a positive integer")
    groups, offset = [], 0
    for index, segment in enumerate(segments):
        rows, length = segment["rows"], segment["sequence_length_used"]
        if type(rows) is not int or type(length) is not int or rows < 1 or length < 1:
            raise ValueError("invalid segment geometry")
        step = max(1, group_tokens // length)
        for first in range(0, rows, step):
            last = min(first + step, rows)
            groups.append({"segment": index, "first_row": first, "last_row": last,
                           "start": offset + first * length, "stop": offset + last * length})
        offset += rows * length
    return groups


def diagnose_receipts(fold: dict, probe: dict) -> dict:
    if probe.get("verdict") != "sealed" or probe.get("problems"):
        raise ValueError("probe must have its authentic completed receipt")
    if fold.get("tool_sha256") != PINS["k3_dense_pass.py"]:
        raise ValueError("fold engine mismatch")
    if probe.get("tool_sha256") != PINS["k3_teacher_cache-gpu6.py"]:
        raise ValueError("cache tool mismatch")
    expected_lane = {name: value for name, value in PINS.items()
                     if name not in ("k3_dense_pass.py", "k3_teacher_cache-gpu6.py")}
    if fold.get("lane_sha256") != expected_lane or probe.get("engine", {}).get("lane_sha256") != expected_lane:
        raise ValueError("lane source mismatch")
    if probe.get("engine", {}).get("sha256") != PINS["k3_dense_pass.py"]:
        raise ValueError("probe engine mismatch")
    integrity = probe.get("integrity", {})
    if not integrity.get("after_equals_before") or integrity.get("changed") or integrity.get("drift_vs_fold_receipt"):
        raise ValueError("recorded source integrity does not match")
    original = fold["corpus"]["segments"]
    current = probe["corpus"]["segments"]
    if len(original) != len(current) or len(original) != len(probe["corpus"]["files"]):
        raise ValueError("corpus segment mismatch")
    for left, right in zip(original, current):
        if (left["rows"], left["sequence_length"]) != (right["rows"], right["sequence_length_used"]):
            raise ValueError("full padded corpus shape mismatch")
    for left, right in zip(original, probe["corpus"]["files"]):
        # Both receipts hash the identical whole safetensors and int64 input IDs.
        # Their real-length helper hashes use different dtype conventions.
        if left["file_sha256"] != right["file_sha256"] or left["input_ids_sha256"] != right["input_ids_sha256"]:
            raise ValueError("corpus file/IDs mismatch")
    equal = []
    for key in sorted(probe["layers"], key=int):
        equal.append({"layer": int(key), "equal": probe["layers"][key]["prefix_out_sha256"] ==
                      fold["layers"][key]["prefix_out_sha256"]["teacher"]})
    first = next((row["layer"] for row in equal if not row["equal"]), None)
    groups = make_groups(current, probe["corpus"]["group_tokens"])
    if sum(item["stop"]-item["start"] for item in groups) != fold["corpus"]["tokens"] or fold["corpus"]["tokens"] != probe["corpus"]["positions"]:
        raise ValueError("full corpus position coverage mismatch")
    return {
        "schema": "chimera_recovery.k3_l2_diagnostic_plan.v1",
        "classification": "unresolved_execution_shape_hypothesis",
        "production_patch_proven": False, "parity_pass": False, "training_admitted": False,
        "first_unequal_layer": first, "layer_comparison": equal,
        "whole_corpus_positions": fold["corpus"]["tokens"],
        "reference_hash_semantics": "SHA256 of contiguous FP32-upcast tensor bytes (original k3_common.sha256_tensor)",
        "grouped_positions": [item["stop"] - item["start"] for item in groups], "groups": groups,
        "source_pins": PINS, "fold_receipt_sha256": FOLD_SHA, "probe_receipt_sha256": PROBE_SHA,
        "unchanged_original_thresholds": True,
        "different_matrix_dimensions": ["router_fp32_linear.M", "selected_expert_w1_w3_linear.M", "selected_expert_w2_linear.M"],
        "runtime_provenance_limits": ["probe does not record backend reduction flags", "fold threads=22; probe threads=16",
                                      "Python patch versions differ", "same sm75 class does not prove same library execution"],
        "next_authentic_diagnostic": {
            "same_input": "one source-pinned teacher residual pair entering L2, prefix and full block stack",
            "arms": ["whole_corpus_repeat_A", "whole_corpus_repeat_B", "five_groups_repeat_A", "five_groups_repeat_B"],
            "api": "replay_shared_pair(engine, runner, pair, corpus, groups, device, row_chunk=1, token_chunk=1024)",
            "taps": ["carried", "ffn_in", "latent_in", "router_topk_idx", "router_topk_w", "y_lat", "prefix_out"],
            "checks": ["input/weight hashes unchanged", "each arm repeat bitwise equality", "whole prefix equals original L2 SHA",
                       "grouped prefix equals original bisect L2 SHA", "first unequal tap and quantitative error"],
            "fallback": "If before_L02 was removed, replay L0/L1 over the SAME full corpus and require both original prefix hashes first.",
            "no_admission": "L2 cause isolation alone cannot pass B2; final unchanged B2 gates still require genuine full-stack recapture.",
        },
    }


def tensor_record(tensor, torch) -> dict:
    value = tensor.detach().contiguous().cpu()
    raw_hash, fold_hash, finite = hashlib.sha256(), hashlib.sha256(), True
    for part in value.reshape(-1).split(65536):
        raw_hash.update(part.view(torch.uint8).numpy().tobytes())
        fold_hash.update(part.float().numpy().tobytes())
        finite = finite and bool(torch.isfinite(part).all())
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": raw_hash.hexdigest(),
            "fold_fp32_sha256":fold_hash.hexdigest(), "finite":finite}


def replay_shared_pair(engine, runner, pair, corpus, groups, device, *, row_chunk=1, token_chunk=1024):
    """Original engine calls on shared parent input; returns measured tap records.

    Caller supplies authentic loaded inputs/weights and seals their provenance.
    This does not execute arbitrary memory commands, fit a child, or declare parity.
    Full tap tensors are kept only within one arm; compare records across arms.
    """
    import torch
    cursor = 0
    for group in groups:
        if group["start"] != cursor or group["stop"] <= cursor:
            raise ValueError("group coverage must be complete, ordered, nonoverlapping")
        cursor = group["stop"]
    if cursor != corpus.tokens or pair.prefix.shape[0] != cursor or pair.blocks.shape[0] != cursor:
        raise ValueError("group/pair corpus coverage mismatch")
    original = [tensor_record(pair.prefix, torch), tensor_record(pair.blocks, torch)]
    outputs = {}

    class View:
        def __init__(self, group):
            self.group = group
            self.tokens = group["stop"] - group["start"]
        def row_chunks(self, chunk):
            group = self.group
            segment = corpus.segments[group["segment"]]
            for first in range(group["first_row"], group["last_row"], chunk):
                last = min(first + chunk, group["last_row"])
                yield segment, first, last, (first-group["first_row"])*segment.T, (last-group["first_row"])*segment.T

    with torch.no_grad():
        for name, grouped in (("whole", False), ("grouped", True)):
            tap_parts = {}
            spans = groups if grouped else [{"start": 0, "stop": corpus.tokens}]
            for span in spans:
                part = engine.Pair(pair.prefix[span["start"]:span["stop"]], pair.blocks[span["start"]:span["stop"]])
                view = View(span) if grouped else corpus
                carried, ffn, _ = engine.attention_half(runner, part, view, device, row_chunk=row_chunk, token_chunk=token_chunk)
                prefix, taps = engine.ffn_half(runner, carried, ffn, runner.cfg, device=device, token_chunk=token_chunk)
                for key, tensor in {"carried": carried, "ffn_in": ffn, "prefix_out": prefix, **{
                        k: taps[k] for k in ("latent_in", "router_topk_idx", "router_topk_w", "y_lat") if k in taps}}.items():
                    tap_parts.setdefault(key, []).append(tensor.detach().cpu())
            outputs[name] = {key: torch.cat(parts) for key, parts in tap_parts.items()}
    if original != [tensor_record(pair.prefix, torch), tensor_record(pair.blocks, torch)]:
        raise RuntimeError("diagnostic changed the supplied residual pair")
    report = {"schema": "chimera_recovery.k3_shared_pair_replay.v1", "parent_parity_proven": False, "input_unchanged": True,
              "whole": {}, "grouped": {}, "comparisons": {}}
    for name, taps in outputs.items():
        report[name] = {key: tensor_record(value, torch) for key, value in taps.items()}
    for key in outputs["whole"]:
        a, b = outputs["whole"][key], outputs["grouped"][key]
        error_sq, norm_sq, maximum, equal = 0.0, 0.0, 0.0, True
        for left,right in zip(a.reshape(-1).split(65536),b.reshape(-1).split(65536)):
            delta=left.double()-right.double()
            error_sq+=delta.square().sum().item();norm_sq+=left.double().square().sum().item()
            maximum=max(maximum,delta.abs().max().item() if delta.numel() else 0.0)
            equal=equal and bool(torch.equal(left.view(torch.uint8),right.view(torch.uint8)))
        report["comparisons"][key] = {"bitwise_equal":equal,"max_abs":maximum,
                                     "relL2":(error_sq/norm_sq)**0.5 if norm_sq else None}
    return report


def load_engine(directory: Path):
    verify_sources(directory)
    root = str(directory.resolve())
    # An already-loaded unverified module cannot masquerade as this capsule.
    for name in ("k3_dense_pass", "k3_common", "k3_layer", "k3_stream", "c14_union_coreset", "k3_receipt", "k3_review_stream"):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent != directory.resolve():
            raise ValueError("different engine already imported: " + name)
    os.environ["K3_LANE_DIR"] = root
    sys.path.insert(0, root)
    return importlib.import_module("k3_dense_pass")


def _cpu_fixture(directory: Path):
    """Small authored packed-MXFP4 weights; never source parent parameters."""
    engine = load_engine(directory)
    import torch
    from k3_layer import LayerWeights
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    hidden, latent, expert_width, experts = 64, 32, 32, 8
    cfg = {"hidden_size": hidden, "rms_norm_eps": 1e-6, "activation_situ_beta": 1.0,
           "activation_situ_linear_beta": 1.0, "num_experts": experts, "num_experts_per_token": 2,
           "moe_renormalize": True, "routed_scaling_factor": 1.0, "moe_router_activation_func": "sigmoid"}
    index = SimpleNamespace(is_kda_layer=lambda _: True, is_moe_layer=lambda _: True, layer_prefix_for=lambda _: "")
    tensors = {}
    shapes = {"gate.weight": (experts, hidden), "routed_expert_down_proj.weight": (latent, hidden),
              "routed_expert_up_proj.weight": (hidden, latent), "shared_experts.gate_proj.weight": (32, hidden),
              "shared_experts.up_proj.weight": (32, hidden), "shared_experts.down_proj.weight": (hidden, 32)}
    for name, shape in shapes.items():
        tensors["block_sparse_moe."+name] = (torch.randn(shape)*0.1).to(torch.bfloat16)
    tensors["block_sparse_moe.gate.e_score_correction_bias"] = torch.linspace(-0.05, 0.05, experts).to(torch.bfloat16)
    tensors["block_sparse_moe.routed_expert_norm.weight"] = torch.ones(latent, dtype=torch.bfloat16)
    for expert in range(experts):
        for name, shape in {"w1": (expert_width, latent), "w3": (expert_width, latent), "w2": (latent, expert_width)}.items():
            prefix = f"block_sparse_moe.experts.{expert}.{name}."
            tensors[prefix+"weight_packed"] = torch.randint(0, 256, (shape[0], shape[1]//2), dtype=torch.uint8)
            tensors[prefix+"weight_scale"] = torch.full((shape[0], shape[1]//32), 122, dtype=torch.uint8)
    weights = LayerWeights(2, tensors, index)
    return engine, torch, cfg, weights, tensors


def cpu_control(directory: Path) -> dict:
    """Authored small packed-MXFP4 MoE control through genuine source operations."""
    engine, torch, cfg, weights, tensors = _cpu_fixture(directory)
    hidden = cfg["hidden_size"]
    runner = SimpleNamespace(w=weights)
    x, carried = torch.randn(257, hidden).to(torch.bfloat16), torch.randn(257, hidden).to(torch.bfloat16)
    receipts, arrays = {}, {}
    weights_before = {key: tensor_record(value, torch) for key, value in tensors.items()}
    with torch.no_grad():
        for arm, spans in [("whole_A", [(0,257)]), ("whole_B", [(0,257)]),
                           ("grouped_A", [(0,64),(64,128),(128,192),(192,257)]),
                           ("grouped_B", [(0,64),(64,128),(128,192),(192,257)])]:
            parts = {}
            for start, stop in spans:
                prefix, taps = engine.ffn_half(runner, carried[start:stop], x[start:stop], cfg,
                                               device=torch.device("cpu"), token_chunk=32)
                for key, tensor in {"prefix_out":prefix, **{k:taps[k] for k in ("latent_in","router_topk_idx","router_topk_w","y_lat")}}.items():
                    parts.setdefault(key, []).append(tensor)
            arrays[arm] = {key:torch.cat(value) for key,value in parts.items()}
            receipts[arm] = {key:tensor_record(value,torch) for key,value in arrays[arm].items()}
    repeats = {name: receipts[name+"_A"] == receipts[name+"_B"] for name in ["whole","grouped"]}
    if not all(repeats.values()) or weights_before != {key:tensor_record(value,torch) for key,value in tensors.items()}:
        raise RuntimeError("repeat or input-weight preservation control failed")
    if not all(rec["finite"] for values in receipts.values() for rec in values.values()):
        raise RuntimeError("authored arithmetic control is nonfinite")
    return {"schema":"chimera_recovery.k3_grouping_cpu_control.v1", "source_pins":PINS,
            "synthetic_operator_authored_2026":True,"real_parent_weights":False,"parent_parity_proven":False,
            "torch":torch.__version__,"device":"cpu","seed":20261013,"repeats_equal":repeats,
            "weights_unchanged":True,"arms":receipts,
            "whole_grouped_bitwise_equal":{key:receipts["whole_A"][key]==receipts["grouped_A"][key] for key in receipts["whole_A"]}}


def cpu_pair_control(directory: Path) -> dict:
    """Exercise shared-pair replay using actual KDA/AttnRes/MoE source functions."""
    engine, torch, cfg, weights, tensors = _cpu_fixture(directory)
    from k3_layer import StreamedLayer
    hidden, attention = cfg["hidden_size"], 32
    cfg.update(attn_res_block_size=12, linear_attn_config={"num_heads":2,"head_dim":16})
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attention_res_norm.weight", "mlp_res_norm.weight"):
        weights.res[name] = torch.ones(hidden,dtype=torch.bfloat16)
    weights.res["self_attention_res_norm.weight"][3] = 0  # valid zero gain, multiplied normally
    for name in ("self_attention_res_proj.weight","mlp_res_proj.weight"):
        weights.res[name] = (torch.randn(1,hidden)*0.1).to(torch.bfloat16)
    for name,shape in {"q_proj.weight":(attention,hidden),"k_proj.weight":(attention,hidden),
                      "v_proj.weight":(attention,hidden),"g_proj.weight":(attention,hidden),
                      "o_proj.weight":(hidden,attention),"f_a_proj.weight":(16,hidden),
                      "f_b_proj.weight":(attention,16),"b_proj.weight":(2,hidden)}.items():
        weights.attn[name] = (torch.randn(shape)*0.1).to(torch.bfloat16)
    for name in ("q_conv1d.weight","k_conv1d.weight","v_conv1d.weight"):
        weights.attn[name] = (torch.randn(attention,1,3)*0.1).to(torch.bfloat16)
    weights.attn.update({"A_log":torch.zeros(16,dtype=torch.bfloat16),
                        "dt_bias":torch.zeros(attention,dtype=torch.bfloat16),
                        "o_norm.weight":torch.ones(16,dtype=torch.bfloat16)})
    runner=StreamedLayer(weights,cfg,kda_backend="torch")
    class Corpus:
        segments=[SimpleNamespace(T=8,rows=3),SimpleNamespace(T=4,rows=2)]
        tokens=32
        def row_chunks(self,chunk):
            offset=0
            for segment in self.segments:
                for first in range(0,segment.rows,chunk):
                    last=min(first+chunk,segment.rows)
                    yield segment,first,last,offset+first*segment.T,offset+last*segment.T
                offset+=segment.rows*segment.T
    corpus=Corpus()
    groups=make_groups([{"rows":3,"sequence_length_used":8},{"rows":2,"sequence_length_used":4}],16)
    pair=engine.Pair(torch.randn(32,hidden).to(torch.bfloat16),torch.randn(32,1,hidden).to(torch.bfloat16))
    before={group:{key:tensor_record(value,torch) for key,value in getattr(weights,group).items()}
            for group in ("res","attn","body")}
    first=replay_shared_pair(engine,runner,pair,corpus,groups,torch.device("cpu"),token_chunk=4)
    second=replay_shared_pair(engine,runner,pair,corpus,groups,torch.device("cpu"),token_chunk=4)
    unchanged=before=={group:{key:tensor_record(value,torch) for key,value in getattr(weights,group).items()}
                      for group in ("res","attn","body")}
    if first!=second or not unchanged:
        raise RuntimeError("shared-pair replay failed repetition/weight preservation")
    return {"schema":"chimera_recovery.k3_shared_pair_cpu_control.v1","synthetic_operator_authored_2026":True,
            "real_parent_weights":False,"parent_parity_proven":False,"torch":torch.__version__,
            "repeats_bitwise_equal":True,"weights_unchanged":unchanged,"record":first}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "cpu-control", "cpu-pair-control"])
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_sources(args.sources)
    if args.out.exists():
        raise ValueError("preserve existing diagnostic output")
    started = time.monotonic()
    if args.command == "cpu-control":
        result = cpu_control(args.sources)
    elif args.command == "cpu-pair-control":
        result = cpu_pair_control(args.sources)
    else:
        fold = args.sources / "k3_dense_pass_receipt.json"
        if file_sha(fold) != FOLD_SHA or not args.probe or file_sha(args.probe) != PROBE_SHA:
            raise ValueError("authentic receipt file pin mismatch")
        result = diagnose_receipts(json.loads(fold.read_text()), json.loads(args.probe.read_text()))
    result["elapsed_seconds"] = time.monotonic()-started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output":str(args.out),"sha256":file_sha(args.out),"classification":result.get("classification"),
                      "parent_parity_proven":False}))


if __name__ == "__main__":
    main()
