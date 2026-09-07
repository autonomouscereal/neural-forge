"""Capture one accepted folded K3 layer, without parent experts or fitting.

This is a new observation tool, NOT a resume-context migration. It imports the
unchanged, receipt-pinned attention_half/ffn_half implementation. Full original
sequences (including review and padding) are replayed; only original evidence
byte-boundary pairs are emitted. Teacher reference verification is explicitly
NOT a teacher forward replay. See C17_K3_CAPTURE.md for the strict plan contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import c17_capture_host as host
import k3_dense_pass as native

SCHEMA = "neural-forge.c17-k3-boundaries.v1"
PLAN_SCHEMA = "neural-forge.c17-k3-capture-plan.v1"
Refusal = host.Refusal
SHARED = tuple("block_sparse_moe." + name for name in (
    "routed_expert_down_proj.weight", "routed_expert_up_proj.weight", "routed_expert_norm.weight",
    "shared_experts.gate_proj.weight", "shared_experts.up_proj.weight", "shared_experts.down_proj.weight"))
BOUNDARIES = ("residual_in", "attention_read", "mixer_input", "mixer_output", "post_attention",
              "ffn_read", "ffn_input", "ffn_output", "residual_out")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def tensor_sha(tensor: torch.Tensor, rows: int = 1024) -> str:
    """Exactly native.sha256_tensor's FP32 byte stream, bounded host memory."""
    h = hashlib.sha256()
    for piece in tensor.detach().split(rows, dim=0):
        h.update(piece.to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


class Pins:
    """Stable full-file verification; paths are explicitly mounted, never inferred."""
    def __init__(self):
        self.records: dict[Path, tuple[tuple[int, ...], str]] = {}

    def file(self, path: str | Path, expected: str) -> Path:
        path = Path(path)
        require(path.is_absolute(), f"input path must be absolute: {path}")
        require(isinstance(expected, str) and bool(host.SHA.fullmatch(expected)), f"invalid SHA256: {path}")
        path = path.resolve(strict=True)
        before = host.stamp(path)
        if path in self.records:
            require(self.records[path] == (before, expected), f"conflicting or changing pin: {path}")
        else:
            require(host.digest(path) == expected and host.stamp(path) == before,
                    f"SHA256 mismatch or changing input: {path}")
            self.records[path] = (before, expected)
        return path

    def ref(self, ref: dict[str, Any]) -> Path:
        return self.file(ref["path"], ref["sha256"])

    def json(self, ref: dict[str, Any]) -> dict[str, Any]:
        path = self.ref(ref)
        value = host.pinned_json(path, ref["sha256"])
        return value

    def stable(self) -> None:
        for path, (before, _sha) in self.records.items():
            require(host.stamp(path) == before, f"input changed during capture: {path}")


def software_identity(pins: Pins) -> dict[str, Any]:
    tool = Path(native.__file__).resolve()
    lane = native.lane_file_digests()
    for name, expected in lane.items():
        path = pins.file(Path(native.LANE_DIR) / name, expected)
        loaded = sys.modules.get(name.removesuffix(".py"))
        if loaded is not None:
            require(Path(loaded.__file__).resolve() == path, f"wrong imported native module: {name}")
    tool_sha = host.digest(tool)
    pins.file(tool, tool_sha)
    packages = {}
    for name in ("flash-linear-attention", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"tool_sha256": tool_sha, "lane_sha256": lane, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "packages": packages}


def runtime_observed(runtime: dict[str, Any]) -> dict[str, Any]:
    """Numerical chunk/dtype settings are consumed directly, not user-overridable."""
    matmul = torch.backends.cuda.matmul
    flags = {key: getattr(matmul, key, None) for key in (
        "allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")}
    for name in ("flash_sdp_enabled", "math_sdp_enabled", "mem_efficient_sdp_enabled", "cudnn_sdp_enabled"):
        getter = getattr(torch.backends.cuda, name, None)
        flags[name] = getter() if getter is not None else None
    flags.update({"cudnn_" + name: getattr(torch.backends.cudnn, name, None)
                  for name in ("enabled", "benchmark", "deterministic", "allow_tf32")})
    return {**runtime, "devices": native.device_inventory(), "backend_flags": flags,
            "num_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def check_corpus(corpus: native.Corpus, accepted: dict[str, Any], rows: list[dict[str, Any]],
                 boundary: dict[str, Any], files: list[Path]) -> dict[str, torch.Tensor]:
    require(corpus.sha256 == accepted["resume_context"]["corpus"], "original whole-sequence corpus identity mismatch")
    original = accepted["corpus"]["segments"]
    require(len(original) == len(corpus.segments), "review segment coverage mismatch")
    for seg, rec, path in zip(corpus.segments, original, files):
        require(host.digest(path) == rec["file_sha256"], "accepted corpus file SHA mismatch")
        require(Path(rec["path"]).name == path.name, "original corpus basename/order must be retained")
        with safe_open(str(path), framework="pt", device="cpu") as sf:
            require(sf.get_slice("input_ids").get_dtype() == "I64", "input_ids must be native I64")
        require(seg.rows > 0 and seg.T > 0 and seg.split is not None, "missing original row groups")
        require(seg.real_lengths.shape == (seg.rows,) and bool(((seg.real_lengths > 0) &
                (seg.real_lengths <= seg.T)).all()), "invalid real sequence lengths")
    first = corpus.segments[0]
    require(first.rows == len(rows) and first.T == boundary["sequence_length"], "evidence row coverage mismatch")
    require(boundary.get("ids_files_sha256", {}).get("k3_input_ids") == host.digest(files[0]),
            "boundary map does not bind these exact K3 evidence IDs")
    mapped = {r["row_index"]: r for r in boundary["rows"] if r.get("file") == "k3_input_ids"}
    vectors = {k: [] for k in ("evidence_row_index", "batch_index", "host_position", "k3_position", "held",
                               "host_flat_position", "k3_flat_position")}
    for row in rows:
        i, split = row["row_index"], int(row["split"] == "held")
        require(int(first.split[i]) == split, "original fit/held split mismatch")
        require(int(first.real_lengths[i]) == mapped[i]["k3_tokens_real"], "K3 real-token boundary mismatch")
        for q, k in row["pairs"]:
            flat = i * first.T + k
            require(bool(corpus.real[flat]) and bool(corpus.held[flat]) == bool(split)
                    and bool(corpus.fit[flat]) != bool(split), "paired row lost original real fit/held mask")
            values = (i, row["batch_index"], q, k, split, i * first.T + q, flat)
            for name, value in zip(vectors, values):
                vectors[name].append(value)
    return {key: torch.tensor(values, dtype=torch.int64) for key, values in vectors.items()}


def acceptance(value: dict[str, Any]) -> bool:
    return (value.get("version") == native.ACCEPTANCE_VERSION and
            value.get("constants") == native.ACCEPTANCE_CONSTANTS and value.get("verdict") == "accepted")


def accepted_chain(plan: dict[str, Any], receipt: dict[str, Any], index: native.K3Index,
                   pins: Pins, layer: int) -> dict[str, str]:
    """Verify every MoE ancestor's actual artifact and done marker, in order."""
    wanted = {str(i) for i in range(layer + 1) if index.is_moe_layer(i)}
    require(set(plan["accepted_layers"]) == wanted, "complete accepted MoE lineage required")
    installed: dict[str, str] = {}
    context = receipt["resume_context"]
    for i in sorted(map(int, wanted)):
        record = receipt["layers"][str(i)]
        refs = plan["accepted_layers"][str(i)]
        sha = record["ffn"]["sha256"]
        require(refs["ffn"]["sha256"] == sha, "FFN pin differs from accepted receipt")
        pins.ref(refs["ffn"])
        done = pins.json(refs["done"])
        expected_lineage = {"version": native.PREFIX_LINEAGE_VERSION, "upstream_ffn_sha256": dict(installed)}
        require(done.get("layer") == i and done.get("moe") is True and done.get("artefact_sha256") == sha,
                "done marker layer/FFN mismatch")
        require(done.get("resume_context") == context and done.get("corpus_sha256") == context["corpus"],
                "done marker source/runtime/corpus mismatch")
        require(done.get("prefix_lineage") == expected_lineage and record.get("prefix_lineage") == expected_lineage,
                "accepted FFN upstream lineage mismatch")
        require(done.get("rung") == receipt["arguments"]["rung"] and
                acceptance(done.get("fold_metrics", {}).get("acceptance", {})) and
                acceptance(record.get("acceptance", {})), "not an accepted production fold")
        recorded_done_sha = record["ffn"].get("done_marker_sha256")
        if recorded_done_sha:
            require(recorded_done_sha == refs["done"]["sha256"], "done marker differs from accepted receipt")
        installed[str(i)] = sha
    return installed


def snapshot(ref: dict[str, Any], manifest: dict[str, Any], pins: Pins, context: dict[str, Any],
             installed: dict[str, str], layer: int, stream: str, rung: int,
             tokens: int, hidden: int, block_size: int) -> tuple[native.Pair, dict[str, Any]]:
    entry = manifest["files"].get(ref["manifest_key"])
    require(isinstance(entry, dict), "explicit snapshot manifest key missing (possibly pruned)")
    lineage = {k: v for k, v in installed.items() if int(k) < layer}
    require(entry.get("layer") == layer and entry.get("stream") == stream and
            entry.get("installed") == lineage and entry.get("corpus_sha256") == context["corpus"] and
            entry.get("rung") == rung, "snapshot manifest lineage/context mismatch")
    path = pins.file(ref["path"], entry["sha256"])
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        meta = sf.metadata() or {}
        require(set(sf.keys()) == {"prefix_sum", "block_residual"}, "unexpected snapshot tensor set")
        require(meta.get("layer") == str(layer) and meta.get("stream") == stream and
                meta.get("run_id") == entry.get("run_id") and meta.get("rung") == str(rung) and
                meta.get("input_ids_sha256") == context["corpus"] and
                meta.get("prefix_lineage_version") == native.PREFIX_LINEAGE_VERSION and
                meta.get("resume_context") == native.canonical_json(context) and
                meta.get("installed_ffn_sha256") == native.canonical_json(lineage) and
                meta.get("installed_layers") == str(len(lineage)), "snapshot metadata identity mismatch")
        blocks = (layer + block_size - 1) // block_size
        require(sf.get_slice("prefix_sum").get_shape() == [tokens, hidden] and
                sf.get_slice("block_residual").get_shape() == [tokens, blocks, hidden] and
                meta.get("blocks") == str(blocks), "snapshot prefix/BlockAttnRes shape mismatch")
        prefix, residuals = sf.get_tensor("prefix_sum"), sf.get_tensor("block_residual")
    require(prefix.dtype in (torch.float16, torch.bfloat16, torch.float32) and residuals.dtype == prefix.dtype,
            "snapshot dtype mismatch (accepted FP32-byte equality must be lossless)")
    pins.stable()
    return native.Pair(prefix, residuals), {"path": str(path), "sha256": entry["sha256"],
        "manifest_key": ref["manifest_key"], "metadata": meta, "prefix_sha256": tensor_sha(prefix),
        "block_residual_sha256": tensor_sha(residuals)}


def load_layer(plan: dict[str, Any], receipt: dict[str, Any], index: native.K3Index,
               pins: Pins, layer: int, device: torch.device) -> tuple[Any, dict[str, torch.Tensor], dict[str, Any]]:
    body_rec = receipt["layers"][str(layer)]["body"]
    require(body_rec.get("byte_exact") is True and plan["body"]["sha256"] == body_rec["sha256"],
            "selected native body is not the accepted verbatim body")
    body_path, ffn_path = pins.ref(plan["body"]), pins.ref(plan["accepted_layers"][str(layer)]["ffn"])
    body, ffn = load_file(str(body_path)), load_file(str(ffn_path))
    prefix = index.layer_prefix_for(layer)
    names = {n.removeprefix(prefix) for n in index.layer_keys(layer)}
    expected_body = {n for n in names if not n.startswith(native.BODY_EXCLUDED_PREFIXES) and ".experts." not in n}
    require(set(body) == expected_body and bool(body), "native body key coverage mismatch")
    require(set(ffn) == {*SHARED, "w1_s", "w3_s", "w2p", "channels"}, "folded FFN contains unexpected/parent expert keys")
    require(set(body).isdisjoint(ffn), "body/FFN key collision")
    with safe_open(str(body_path), framework="pt") as sf:
        require((sf.metadata() or {}).get("kind") == "non_expert_body_verbatim" and
                (sf.metadata() or {}).get("layer") == str(layer), "native body metadata mismatch")
    with safe_open(str(ffn_path), framework="pt") as sf:
        meta = sf.metadata() or {}
        require(meta.get("form") == "E" and meta.get("layer") == str(layer) and
                meta.get("input_ids_sha256") == receipt["resume_context"]["corpus"], "FFN metadata mismatch")
    h, p = index.hidden_size, int(index.text_config["routed_expert_hidden_size"])
    width = ffn["w1_s"].shape[0]
    require(ffn["w1_s"].shape == (width, p) and ffn["w3_s"].shape == (width, p) and
            ffn["w2p"].shape == (p, width) and ffn["channels"].shape == (width, 2) and
            meta.get("hidden") == str(h) and meta.get("latent_dim") == str(p) and
            meta.get("width_installed") == str(width), "form-E dimension mismatch")
    require(all(t.is_floating_point() for n, t in {**body, **ffn}.items() if n != "channels"),
            "non-floating native weight")
    require(ffn["channels"].dtype in (torch.int32, torch.int64) and bool((ffn["channels"] >= 0).all()) and
            bool((ffn["channels"][:, 0] < index.text_config["num_experts"]).all()) and
            bool((ffn["channels"][:, 1] < index.text_config["moe_intermediate_size"]).all()),
            "invalid accepted channel bookkeeping")
    weights = native.LayerWeights(layer, {**body, **{k: ffn[k] for k in SHARED}}, index)
    require(not weights.experts, "parent expert loading is forbidden")
    dtype, why = native.resolve_compute_dtype(weights, native.torch_dtype(receipt["resume_context"]["runtime"]["compute_dtype"]))
    source_dtypes = {k: str(t.dtype) for k, t in {**body, **ffn}.items()}
    weights.to(device)
    for group in (weights.attn, weights.res, weights.body):
        for key, value in group.items():
            if value.is_floating_point():
                group[key] = value.to(dtype)
    dense = {k: ffn[k].to(device) for k in ("w1_s", "w3_s", "w2p")}
    pins.stable()
    return native.StreamedLayer(weights, index.text_config,
                               kda_backend=receipt["resume_context"]["runtime"]["kda_backend"]), dense, {
        "source_dtypes": source_dtypes, "resolved_compute_dtype": str(dtype), "dtype_resolution": why,
        "body_sha256": body_rec["sha256"], "parent_experts_loaded": False}


class Observe:
    """Scoped observation wrappers call native functions once and return untouched results."""
    def __init__(self, runner: Any, corpus: native.Corpus, positions: torch.Tensor):
        self.runner, self.corpus, self.positions = runner, corpus, positions
        self.tokens = corpus.tokens
        self.rows: dict[str, torch.Tensor] = {}
        self.seen: dict[str, torch.Tensor] = {}
        self.chunk = (0, 0)

    def take(self, name: str, value: torch.Tensor, start: int = 0) -> None:
        value = value.reshape(-1, value.shape[-1])
        mask = (self.positions >= start) & (self.positions < start + value.shape[0])
        indices = mask.nonzero().flatten()
        if indices.numel() == 0:
            return
        if name not in self.rows:
            self.rows[name] = torch.empty(len(self.positions), value.shape[-1], dtype=value.dtype)
            self.seen[name] = torch.zeros(len(self.positions), dtype=torch.bool)
        require(not bool(self.seen[name][indices].any()), f"duplicate observed boundary: {name}")
        selected = value.index_select(0, (self.positions[indices] - start).to(value.device)).detach().cpu()
        require(bool(torch.isfinite(selected).all()), f"nonfinite observed boundary: {name}")
        self.rows[name][indices] = selected
        self.seen[name][indices] = True

    def row_chunks(self, row_chunk: int):
        for record in self.corpus.row_chunks(row_chunk):
            self.chunk = record[-2:]
            yield record

    def __enter__(self):
        self.original_attention = self.runner.attention
        self.original_res = native.apply_attn_res
        self.original_write = native.moe_write

        def attention(x, state=None):
            self.take("mixer_input", x, self.chunk[0])
            result = self.original_attention(x, state=state)
            self.take("mixer_output", result[0], self.chunk[0])
            return result

        def residual(p, blocks, proj, norm, *args, **kwargs):
            result = self.original_res(p, blocks, proj, norm, *args, **kwargs)
            if proj is self.runner.w.res["self_attention_res_proj.weight"]:
                name = "attention_read"
            elif proj is self.runner.w.res["mlp_res_proj.weight"]:
                name = "ffn_read"
            else:
                raise Refusal("unexpected BlockAttnRes projection")
            self.take(name, result, self.chunk[0])
            return result

        def write(*args, **kwargs):
            result = self.original_write(*args, **kwargs)
            self.take("ffn_output", result)
            return result

        self.runner.attention = attention
        native.apply_attn_res = residual
        native.moe_write = write
        return self

    def __exit__(self, *_args):
        self.runner.attention = self.original_attention
        native.apply_attn_res = self.original_res
        native.moe_write = self.original_write


@torch.no_grad()
def replay(runner: Any, dense: dict[str, torch.Tensor], pair: native.Pair, corpus: native.Corpus,
           vectors: dict[str, torch.Tensor], runtime: dict[str, Any]) -> tuple[native.Pair, dict[str, torch.Tensor]]:
    device = torch.device(runtime["device"])
    observer = Observe(runner, corpus, vectors["k3_flat_position"])
    observer.take("residual_in", pair.prefix)
    if pair.blocks.shape[1] == 0:
        observer.take("attention_read", pair.prefix)
    with observer:
        carried, x, blocks = native.attention_half(runner, pair, observer, device,
                row_chunk=runtime["row_chunk"], token_chunk=runtime["token_chunk"])
        observer.take("post_attention", carried)
        observer.take("ffn_input", x)
        latent = native._chunked_linear(x, runner.w.body[SHARED[0]], runtime["token_chunk"])
        prefix, taps = native.ffn_half(runner, carried, x, runner.cfg, device=device,
                token_chunk=runtime["token_chunk"], dense=dense, latent=latent)
        observer.take("residual_out", prefix)
        result = native.Pair(prefix.to("cpu"), blocks)
        del carried, x, latent, taps, prefix
    require(set(observer.rows) == set(BOUNDARIES) and all(bool(x.all()) for x in observer.seen.values()),
            "incomplete native boundary capture")
    return result, {**observer.rows, **vectors}


def run(plan_path: Path, expected_plan_sha: str, out: Path, *, synthetic_fixture_root: Path | None = None) -> dict[str, Any]:
    pins = Pins()
    plan = pins.json({"path": str(plan_path), "sha256": expected_plan_sha})
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported capture plan")
    require(not out.exists(), "output must be fresh; existing/partial captures are never overwritten")
    receipt = pins.json(plan["accepted_receipt"])
    context = receipt["resume_context"]
    require(context.get("version") == native.RESUME_CONTEXT_VERSION, "unsupported native resume identity")
    require(software_identity(pins) == context["software"], "accepted native source/software identity mismatch")
    for module in (Path(__file__).resolve(), Path(host.__file__).resolve()):
        pins.file(module, host.digest(module))
    metadata_dir = Path(plan["model_metadata_directory"])
    pins.file(metadata_dir / "config.json", context["model"]["config_sha256"])
    pins.file(metadata_dir / "model.safetensors.index.json", context["model"]["index_sha256"])
    index = native.K3Index(str(metadata_dir))  # metadata only; no read_keys or parent shard path is used
    runtime = context["runtime"]
    torch.set_num_threads(host.integer(runtime["num_threads"], "num_threads", minimum=1))
    require(runtime_observed(runtime) == runtime, "accepted runtime/device/backend identity mismatch")
    require(runtime["resid_device"] == "cpu" and runtime["kda_backend"] == "torch", "only original CPU-residual/torch lane supported")
    for key in ("row_chunk", "token_chunk"):
        host.integer(runtime[key], key, minimum=1)
    fixture = synthetic_fixture_root is not None
    if fixture:
        root = synthetic_fixture_root.resolve(strict=True)
        require(plan.get("fixture") is True and index.config_json.get("synthetic") is True and
                index.hidden_size <= 128 and index.num_layers <= 8 and runtime["device"] == "cpu",
                "synthetic fixture mode requires a tiny explicitly synthetic CPU model")
    else:
        require(plan.get("fixture") is False and index.config_json.get("synthetic") is not True,
                "synthetic inputs are not production evidence")
        require(runtime["device"].startswith("cuda:"), "production replay requires the accepted GPU runtime")
    layer = host.integer(plan["layer"], "layer", minimum=1)
    require(layer < index.num_layers and index.is_moe_layer(layer), "this canary captures one accepted MoE layer")
    print(f"[c17-k3] L{layer:02d}: source/runtime verified; checking accepted FFN lineage", flush=True)
    installed = accepted_chain(plan, receipt, index, pins, layer)
    files = [pins.ref(plan["input_ids"])] + [pins.ref(ref) for ref in plan["review_input_ids"]]
    corpus = native.Corpus(str(files[0]), [str(p) for p in files[1:]])
    evidence, boundary = pins.json(plan["evidence"]), pins.json(plan["boundary_map"])
    rows = host.evidence_rows(evidence, boundary, plan["evidence"]["sha256"])
    vectors = check_corpus(corpus, receipt, rows, boundary, files)
    if fixture:
        require(corpus.tokens <= 4096 and all(p.is_relative_to(root) for p in [out.resolve(), metadata_dir.resolve(),
                *[p for p in pins.records if p.suffix in (".json", ".safetensors")]]), "fixture inputs/output must stay inside tiny fixture root")
    max_bytes = host.integer(plan.get("max_capture_bytes"), "max_capture_bytes", minimum=1)
    projected = len(vectors["held"]) * (len(BOUNDARIES) * index.hidden_size * 4 + len(vectors) * 8)
    require(projected <= max_bytes, "capture output budget exceeded; no implicit subsampling")
    print(f"[c17-k3] original corpus verified: {corpus.tokens} tokens; {len(vectors['held'])} paired rows; "
          f"capture upper bound {projected} bytes; hashing frozen snapshots", flush=True)
    manifest = pins.json(plan["snapshot_manifest"])
    snap_args = (manifest, pins, context, installed)
    end_args = (receipt["arguments"]["rung"], corpus.tokens, index.hidden_size, index.attn_res_block_size)
    pair, start_record = snapshot(plan["start_folded"], *snap_args, layer, "folded", *end_args)
    previous = receipt["layers"][str(layer - 1)]["prefix_out_sha256"]["folded"]
    require(start_record["prefix_sha256"] == previous, "starting folded prefix differs from accepted upstream receipt")
    teacher, teacher_record = snapshot(plan["end_teacher"], *snap_args, layer + 1, "teacher", *end_args)
    require(teacher_record["prefix_sha256"] == receipt["layers"][str(layer)]["prefix_out_sha256"]["teacher"],
            "ending teacher reference differs from accepted receipt")
    del teacher
    end, end_record = snapshot(plan["end_folded"], *snap_args, layer + 1, "folded", *end_args)
    require(end_record["prefix_sha256"] == receipt["layers"][str(layer)]["prefix_out_sha256"]["folded"],
            "ending folded reference differs from accepted receipt")
    expected_dtype = end.prefix.dtype
    del end
    runner, dense, loading = load_layer(plan, receipt, index, pins, layer, torch.device(runtime["device"]))
    require(pair.prefix.dtype == expected_dtype == next(iter(runner.w.attn.values())).dtype,
            "snapshot/native compute dtype mismatch")
    if fixture:
        require(all(p.is_relative_to(root) for p in pins.records if p.suffix in (".json", ".safetensors")),
                "fixture snapshots/body must stay inside fixture root")
    pins.stable()
    print(f"[c17-k3] snapshots and native body verified; replaying L{layer:02d} with no parent experts", flush=True)
    result, payload = replay(runner, dense, pair, corpus, vectors, runtime)
    require(tensor_sha(result.prefix) == end_record["prefix_sha256"] and
            tensor_sha(result.blocks) == end_record["block_residual_sha256"], "accepted folded prefix/BlockAttnRes replay is NOT exact")
    pins.stable()
    print("[c17-k3] folded prefix and BlockAttnRes exactly match accepted reference; writing observed boundaries", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    folder = out / f"L{layer:02d}"
    folder.mkdir()
    target = folder / "k3_boundaries.safetensors"
    save_file({k: v.contiguous() for k, v in payload.items()}, str(target),
              metadata={"schema": SCHEMA, "layer": str(layer), "stream": "folded", "fixture": str(fixture).lower()})
    record = {"schema": SCHEMA, "fixture": fixture, "stream": "folded", "gradients_or_training": False,
        "accepted_prefix_replay_exact": True, "teacher_reference_verified": True, "teacher_forward_replayed": False,
        "folded_child_proven": False, "parent_experts_loaded": False, "new_fit": False,
        "installed_ffn_sha256": installed, "accepted_receipt": plan["accepted_receipt"],
        "accepted_resume_context": context, "capture_plan": {"path": str(plan_path), "sha256": expected_plan_sha},
        "source_identity_scope": "accepted config/index identity plus hash-verified verbatim body and folded FFNs; no full parent shard rehash",
        "code_sha256": {str(p): sha for p, (_stat, sha) in pins.records.items() if p.suffix == ".py"},
        "evidence_sha256": plan["evidence"]["sha256"], "boundary_sha256": plan["boundary_map"]["sha256"],
        "boundary_seal": boundary["boundary_map_sha256"], "corpus_sha256": corpus.sha256,
        "full_corpus_replayed": True, "corpus": corpus.record(), "position_sequence_length": boundary["sequence_length"],
        "snapshots": {"start_folded": start_record, "end_folded": end_record, "end_teacher": teacher_record},
        "snapshot_manifest": plan["snapshot_manifest"], "loading": loading, "runtime": runtime,
        "boundaries": {"attention_read": "native self_attention BlockAttnRes output before input_layernorm",
            "mixer_input": "actual native KDA/MLA input after input_layernorm; full original row position zero/state=None",
            "mixer_output": "actual native KDA/MLA write", "post_attention": "native carried accumulator (not ffn_read)",
            "ffn_read": "native mlp BlockAttnRes output before post_attention_layernorm",
            "ffn_input": "actual post_attention_layernorm output consumed by routed down projection and shared FFN",
            "ffn_output": "actual native moe_write (dense form-E + verbatim shared experts), before residual addition",
            "residual_in": "folded prefix_sum entering layer", "residual_out": "folded prefix_sum leaving layer"},
        "layers": [{"layer": layer, "kind": "kda" if runner.w.is_kda else "mla", "rows": len(vectors["held"]),
            "path": str(target.relative_to(out)).replace("\\", "/"), "sha256": host.digest(target), "bytes": target.stat().st_size,
            "boundaries": {k: {"shape": list(payload[k].shape), "dtype": str(payload[k].dtype)} for k in BOUNDARIES}}],
        "input_stat_stable": True, "source_host_pairs_complete": False,
        "missing": ["join actual native host captures", "fit and independently validate boundary-domain charts", "all-source child capability demonstration"]}
    pins.stable()
    native.write_receipt(str(out / "capture_receipt.json"), record)
    print(f"[c17-k3] capture receipt written: {out / 'capture_receipt.json'}", flush=True)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-fixture-root", type=Path, help="tiny synthetic CPU tests only; never production evidence")
    args = parser.parse_args(argv)
    run(args.capture_plan, args.plan_sha256, args.out, synthetic_fixture_root=args.synthetic_fixture_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
