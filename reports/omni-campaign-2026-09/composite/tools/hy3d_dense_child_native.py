"""Typed dense-child installation, independent native forward and sealed reload.

Native teacher helpers remain unchanged. Candidate inference intentionally uses
one dense FFN; the original full block forward still owns attention/time/skip.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
import torch
from safetensors import safe_open
import c16_capture_hy3d as capture
import c16_replay_hy3d as historical
import hy3d_full21_native as teacher
from hy3d_full21_linalg import finite, scores
from hy3d_full21_contract import block_contract, require
from hy3d_dense_child_contract import SCHEMA, STATE_NAMES, validate_child


class DenseChild(torch.nn.Module):
    def __init__(self, state, width, keep, device="cpu"):
        super().__init__()
        require(set(state) == STATE_NAMES, "exactly four dense-child tensors required")
        shapes = {"fc1.weight": (keep, width), "fc1.bias": (keep,), "fc2.weight": (width, keep), "fc2.bias": (width,)}
        for name, value in state.items():
            require(tuple(value.shape) == shapes[name] and value.dtype == torch.float16, "dense-child shape/dtype mismatch: " + name)
            finite(value, name)
        with torch.device("meta"):
            self.fc1 = torch.nn.Linear(width, keep, bias=True)
            self.fc2 = torch.nn.Linear(keep, width, bias=True)
        self.gelu = torch.nn.GELU(approximate="none")
        self.requires_grad_(False)
        self.load_state_dict(state, strict=True, assign=True)
        self.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
        require(not any(t.is_meta for t in (*self.parameters(), *self.buffers())), "unmaterialized dense-child tensor")

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

    @classmethod
    def meta(cls, width, keep):
        child = cls.__new__(cls); torch.nn.Module.__init__(child)
        with torch.device("meta"):
            child.fc1 = torch.nn.Linear(width, keep, bias=True)
            child.fc2 = torch.nn.Linear(keep, width, bias=True)
            child.gelu = torch.nn.GELU(approximate="none")
        child.requires_grad_(False)
        return child


@contextmanager
def installed_child(block, index, state, contract, device):
    validate_child(contract)
    require(contract["block_index"] == index, "dense child belongs to another block")
    original = block.moe if index >= 15 else block.mlp
    child = DenseChild(state, contract["width"], contract["keep"], device)
    require(block.use_moe == (index >= 15) and (block.skip_linear is not None) == contract["skip_required"], "source block variant changed")
    name = "moe" if index >= 15 else "mlp"
    setattr(block, name, child)
    try:
        yield child
    finally:
        setattr(block, name, original)


@torch.inference_mode()
def replay_child(block, time, embedding, tensors, row_index, index, device):
    topology = block_contract(index)
    row = {name: tensor[row_index:row_index + 1].to(device) for name, tensor in tensors.items()}
    observed = {}
    def put(name, value):
        require(name not in observed, "repeated dense-child boundary")
        finite(value, "candidate " + name); observed[name] = capture.cpu(value)
    ffn = block.moe if topology["use_moe"] else block.mlp
    require(isinstance(ffn, DenseChild) and not block.training and not any(p.requires_grad for p in block.parameters()), "frozen explicit dense child required")
    with ExitStack() as hooks:
        for name, module in (("self_attention_write", block.attn1), ("cross_attention_write", block.attn2), ("mlp_output", ffn)):
            hooks.callback(module.register_forward_hook(lambda m, a, out, name=name: put(name, out)).remove)
        hooks.callback(ffn.register_forward_pre_hook(lambda m, a: put("mlp_input", a[0])).remove)
        if topology["skip_required"]:
            hooks.callback(block.skip_linear.register_forward_pre_hook(lambda m, a: put("skip_concat", a[0])).remove)
        token = time(row["sigma"], condition=None); put("time_token", token)
        if index == 0:
            put("initial_block_input", torch.cat((token, embedding(row["dit_latent_input"])), dim=1))
        put("block_output", block(row["block_input"], token, row["context"], skip_value=row.get("skip_value")))
    references = {name: tensors[name][row_index:row_index + 1] for name in observed if name not in ("initial_block_input", "skip_concat")}
    if index == 0: references["initial_block_input"] = tensors["block_input"][row_index:row_index + 1]
    if topology["skip_required"]:
        references["skip_concat"] = torch.cat((row["skip_value"], row["block_input"]), dim=-1).cpu()
    metrics = {name: {"byte_exact": teacher.bit_exact(value, references[name]),
               **scores(value.reshape(-1, value.shape[-1]), references[name].reshape(-1, value.shape[-1]))}
               for name, value in observed.items()}
    return observed, metrics


def native_route_masses(tensors, fit_ids):
    result, diagnostics = [], []
    for i in fit_ids:
        ids, weights, logits = (tensors[name][i].reshape(-1, tensors[name].shape[-1])
                              for name in ("router_topk_ids", "router_topk_weights", "router_logits"))
        require(ids.dtype == torch.int64 and weights.dtype == logits.dtype == torch.float16 and ids.shape[1] == 2 and logits.shape[1] == 8,
                "authentic native route storage/geometry required")
        require(bool(((ids >= 0) & (ids < 8)).all()) and bool((ids[:, 0] != ids[:, 1]).all()), "invalid native top2 expert identity")
        probability = logits.softmax(-1)
        # Sealed file/body identity and original gate.norm_topk_prob=False bind
        # semantics. Cross-device F16 softmax reconstruction is a diagnostic,
        # not a numeric refusal; use original captured masses without rewriting.
        require(bool(((weights >= 0) & (weights <= 1)).all()), "invalid native route mass range")
        masses = torch.zeros(len(ids), 9, dtype=torch.float16)
        masses.scatter_(1, ids, weights); masses[:, 8] = 1
        diagnostics.append({"fit_row_index": i, "cpu_softmax_weights_byte_exact": teacher.bit_exact(weights, probability.gather(-1, ids)),
                            "cpu_top2_weights_byte_exact": teacher.bit_exact(weights.sort(-1).values, probability.topk(2, dim=-1).values.sort(-1).values),
                            "numeric_veto": False, "stored_native_masses_rewritten": False})
        result.append(masses)
    return result, diagnostics


def load_child(receipt_path, source, capture_body, rows):
    path = Path(receipt_path).resolve(strict=True)
    raw = capture.bounded_bytes(path, 16 << 20)
    receipt = capture.unique_json(raw)
    require(receipt.get("schema") == SCHEMA and receipt.get("source_identity") == source["source_identity"] and
            receipt.get("block_index") == capture_body["block_index"] and receipt.get("capture_sha256") == capture_body["file_sha256"] and
            receipt.get("fit_object_sha256") == sorted(row["object_sha256"] for row in rows if row["split"] == "fit"), "saved child provenance mismatch")
    contract = validate_child(receipt.get("dense_child"))
    require(contract["block_index"] == receipt["block_index"] and receipt.get("held_object_sha256") ==
            sorted(row["object_sha256"] for row in rows if row["split"] == "held") and
            receipt.get("actual_source_forward_completed") is True and receipt.get("stored_tensor_readback_verified") is True and
            receipt.get("candidate_written") is True and receipt.get("fixture") is False, "saved child actual source/held/readback contract incomplete")
    artifact = historical.sealed_path(path.parent, receipt.get("file"))
    require(capture.tensor_header(artifact)["bytes"] <= 256 << 20 and capture.digest_file(artifact) == receipt.get("file_sha256"), "dense-child hash/bound mismatch")
    with safe_open(str(artifact), framework="pt", device="cpu") as handle:
        require(set(handle.keys()) == STATE_NAMES and set(receipt.get("tensor_identity", {})) == STATE_NAMES, "dense-child tensor inventory incomplete")
        state = {name: handle.get_tensor(name) for name in STATE_NAMES}
    require(all(teacher.tensor_identity(value) == receipt["tensor_identity"][name] for name, value in state.items()), "corrupt dense-child tensor body")
    DenseChild(state, contract["width"], contract["keep"])
    return state, receipt, {path: hashlib.sha256(raw).hexdigest(), artifact: receipt["file_sha256"]}
