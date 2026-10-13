"""Source-sealed authored ShapeVAE supervision, separate from guided HY3D KD.

CPU preparation only. The existing raw parent head has sign -1. Its input time
is native progress (1-canonical sigma), never the canonical corruption scale.
Source XYZ/clean queries/assistant geometry metadata are target-only evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

try:
    from . import geometry_targets as G
    from . import geometry_bridge_probe as P
except ImportError:
    import geometry_targets as G
    import geometry_bridge_probe as P
A, C = G.A, G.C
HERE = Path(__file__).resolve().parent
STAGE = "D3b_geometry_supervised_flow"
OBJECTIVE = "supervised_geometry_rectified_flow"
POLICY = "hy3d_position_free_bridge_native_progress_2026-10-13.v1"
SCHEMA = "recovery_geometry_flow_batches.v1"
AMENDMENT_SCHEMA = "recovery_geometry_flow_amendment.v1"
SELECTION_SCHEMA = "recovery_geometry_flow_selection.v1"
SIGMAS = [.05, .3, .7, 1.]
SEED = 20261013
IMAGE_ROWS, LATENT_ROWS, WIDTH = 1024, 4096, 6144
MEDIA_ROWS = IMAGE_ROWS + 1 + LATENT_ROWS
PLACEHOLDER, TEXT_VOCAB = 248832, 248320
TOKENIZER_PINS = {
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}
COMPACT = {"schema": "recovery_geometry_flow_bands.v1", "width": WIDTH,
           "image_feature": [640, 2176], "geometry_feature": [576, 640],
           "geometry_emission": [320, 384], "time_row": IMAGE_ROWS}


def dependency_pins():
    return {name: A.file_sha(HERE / name) for name in (
        "geometry_targets.py", "geometry_bridge_probe.py", "geometry_conditioning.py",
        "geometry_prepare.py", "hy3d_target_session.py", "hy3d_surface_probe.py", "audio_admission.py")}


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": A.file_sha(path)}


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_messages(record):
    messages = record.get("messages", [])
    if (record.get("schema") != "recovery_corpus.v1" or record.get("route") != "volume_generate" or
        [m.get("role") for m in messages] != ["user", "assistant"] or record.get("tools") or
        record.get("supervision", {}).get("message_mask") not in ([0, 1], [False, True]) or
        any(m.get("tool_calls") for m in messages) or
        record.get("geometry_target", {}).get("generation_query_xyz_allowed") is not False):
        raise ValueError("exact authored volume source and assistant-only target provenance required")
    text = messages[0].get("content")
    if (not isinstance(text, str) or not text.strip() or
        any(s in text for s in ("<|im_start|>", "<|im_end|>"))):
        raise ValueError("unambiguous original user instruction required")
    # Do not derive a caption from clean mesh, query XYZ or assistant metadata.
    return [{"role": "user", "content": text}]


def token_ids(value):
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("one authentic parent prefix required")
        value = value[0]
    if not isinstance(value, list) or not value or any(type(n) is not int or not 0 <= n < TEXT_VOCAB for n in value):
        raise ValueError("native text-only parent token IDs required")
    return value


def pack_prefix(record, tokenizer, *, max_length=8192):
    messages = source_messages(record)
    options = dict(add_generation_prompt=True, enable_thinking=False, tools=None)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    ids = token_ids(tokenizer.apply_chat_template(messages, tokenize=True, **options))
    if (not isinstance(rendered, str) or ids != token_ids(tokenizer(rendered, add_special_tokens=False)) or
        not getattr(tokenizer, "chat_template", None)):
        raise ValueError("exact genuine parent generation-prefix rendering and IDs required")
    if type(max_length) is not int or len(ids) + MEDIA_ROWS > max_length:
        raise ValueError("full image/time/geometry exceeds context; truncation is forbidden")
    return {"schema": "recovery_geometry_generation_prefix.v1", "source_row_sha256": A.digest(record),
            "messages": messages, "parent_text_ids": ids, "prompt_length": len(ids),
            "input_ids": ids + [PLACEHOLDER] * MEDIA_ROWS,
            "media_indices": list(range(len(ids), len(ids) + MEDIA_ROWS)),
            "template_sha256": A.digest(tokenizer.chat_template),
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "native_text_projection_bitwise_equal": True, "assistant_target_in_prefix": False}


def validate_prefix(record, prefix):
    ids = token_ids(prefix.get("parent_text_ids"))
    length = len(ids)
    if (prefix.get("schema") != "recovery_geometry_generation_prefix.v1" or
        prefix.get("source_row_sha256") != A.digest(record) or prefix.get("messages") != source_messages(record) or
        prefix.get("prompt_length") != length or prefix.get("input_ids") != ids + [PLACEHOLDER] * MEDIA_ROWS or
        prefix.get("media_indices") != list(range(length, length + MEDIA_ROWS)) or
        prefix.get("native_text_projection_bitwise_equal") is not True or prefix.get("assistant_target_in_prefix") is not False or
        any(not A.SHA.fullmatch(str(prefix.get(k))) for k in ("template_sha256", "rendered_sha256"))):
        raise ValueError("authentic source-sealed complete generation prefix changed")
    return length


def noise_seed(source_sha, sigma):
    if not A.SHA.fullmatch(str(source_sha)) or type(sigma) not in (int, float) or sigma not in SIGMAS:
        raise ValueError("declared source hash and prescribed positive corruption scale required")
    h = A.digest({"source_row_sha256": source_sha, "sigma": sigma, "seed": SEED,
                  "policy": "source_row_sigma_sha256_first63bits_v1"})
    return int(h[:16], 16) & ((1 << 63) - 1)


def image_arrays(image):
    if set(image) != {"patch_flat", "grid_thw", "yx", "native_media_positions"}:
        raise ValueError("only authentic raw C11 pixel/grid/position arrays allowed")
    pixels = C.unpack(image["patch_flat"], image["grid_thw"])
    if (image["patch_flat"].shape != (IMAGE_ROWS, 1536) or image["patch_flat"].dtype != np.float32 or
        not np.isfinite(pixels).all() or np.any(np.abs(pixels) > 1) or
        not np.array_equal(pixels[0], pixels[1]) or
        image["grid_thw"].dtype != np.int64 or image["yx"].dtype != np.int64 or
        image["native_media_positions"].dtype != np.float32 or
        not np.array_equal(image["grid_thw"], np.array([[1, 32, 32]], dtype=np.int64))):
        raise ValueError("complete finite duplicated-frame C11 source pixels required")
    expected = C.patchify(pixels[0])
    if any(not np.array_equal(image[k], value) for k, value in zip(
            ("patch_flat", "grid_thw", "yx", "native_media_positions"), expected)):
        raise ValueError("original merge-window grid/source image coordinates differ")


def build_batch(record, prefix, image, clean, *, sigma):
    import torch
    text = validate_prefix(record, prefix)
    image_arrays(image)
    if (type(sigma) not in (int, float) or sigma not in SIGMAS or
        tuple(clean.shape) != (1, LATENT_ROWS, 64) or clean.dtype != torch.float32 or clean.requires_grad or
        clean.device.type != "cpu" or not bool(torch.isfinite(clean).all())):
        raise ValueError("detached finite genuine model-space4096x64 CPU target and prescribed sigma required")
    seed = noise_seed(A.digest(record), sigma)
    epsilon = torch.randn(clean.shape, dtype=torch.float32, generator=torch.Generator(device="cpu").manual_seed(seed))
    noisy = (1 - sigma) * clean + sigma * epsilon
    features = torch.zeros(1, MEDIA_ROWS, WIDTH)
    targets = torch.zeros_like(features)
    features[0, :IMAGE_ROWS, 640:2176] = torch.from_numpy(image["patch_flat"].copy())
    features[:, IMAGE_ROWS + 1:, 576:640] = noisy
    targets[:, IMAGE_ROWS + 1:, 320:384] = epsilon - clean
    positions = torch.zeros(1, MEDIA_ROWS, 4)
    positions[0, :IMAGE_ROWS] = torch.from_numpy(image["native_media_positions"].copy())
    positions[:, IMAGE_ROWS] = -1
    levels = torch.zeros(1, MEDIA_ROWS)
    levels[:, IMAGE_ROWS:] = 1 - sigma
    canonical = torch.zeros_like(levels)
    canonical[:, IMAGE_ROWS:] = sigma
    weights = torch.zeros_like(levels)
    weights[:, IMAGE_ROWS + 1:] = 1
    # Complete-sequence *categorical observation* metadata, not physical XYZ.
    # HY3D source currently has no spatial RoPE; H3 owns no type2/3 rows here.
    source_positions = torch.zeros(1, text + MEDIA_ROWS, 3)
    source_positions[0, :text, 0] = torch.arange(text, dtype=torch.float32)
    source_positions[0, text:text + IMAGE_ROWS] = positions[0, :IMAGE_ROWS, :3]
    source_positions[:, text + IMAGE_ROWS] = -1
    return {"input_ids": torch.tensor([prefix["input_ids"]], dtype=torch.int64),
            "native_media_features": features, "native_media_targets": targets,
            "native_media_indices": torch.tensor([prefix["media_indices"]], dtype=torch.int64),
            "native_media_type_ids": torch.tensor([[6] * IMAGE_ROWS + [7] * (LATENT_ROWS + 1)], dtype=torch.int64),
            "native_media_valid_mask": torch.ones(1, MEDIA_ROWS, dtype=torch.bool),
            "native_media_positions": positions, "native_media_noise_levels": levels,
            "native_media_canonical_sigma": canonical, "native_media_loss_weights": weights,
            "native_media_loss_mask": weights.clone(), "charted_source_positions": source_positions}


def compact(batch):
    rest = {k: v.contiguous() for k, v in batch.items() if k not in ("native_media_features", "native_media_targets")}
    f, t = batch["native_media_features"], batch["native_media_targets"]
    rest["image_pixels"] = f[:, :IMAGE_ROWS, 640:2176].contiguous()
    rest["geometry_noisy"] = f[:, IMAGE_ROWS + 1:, 576:640].contiguous()
    rest["geometry_canonical_velocity"] = t[:, IMAGE_ROWS + 1:, 320:384].contiguous()
    # The subtraction checks prevent quietly discarding unallocated content.
    check_f, check_t = f.clone(), t.clone()
    check_f[:, :IMAGE_ROWS, 640:2176] = 0
    check_f[:, IMAGE_ROWS + 1:, 576:640] = 0
    check_t[:, IMAGE_ROWS + 1:, 320:384] = 0
    if bool(check_f.any()) or bool(check_t.any()):
        raise ValueError("geometry compact storage would discard unallocated input or target")
    return rest


def inflate(stored):
    import torch
    arrays = {k: stored[k] for k in ("image_pixels", "geometry_noisy", "geometry_canonical_velocity")}
    for key, shape in (("image_pixels", (1, IMAGE_ROWS, 1536)), ("geometry_noisy", (1, LATENT_ROWS, 64)),
                       ("geometry_canonical_velocity", (1, LATENT_ROWS, 64))):
        value = arrays[key]
        if value.shape != shape or value.dtype != torch.float32 or value.requires_grad or not bool(torch.isfinite(value).all()):
            raise ValueError("compact source pixel/noisy/velocity layout differs")
    batch = {k: v for k, v in stored.items() if k not in arrays}
    if "native_media_features" in batch or "native_media_targets" in batch:
        raise ValueError("duplicate dense native content in compact geometry storage")
    features = torch.zeros(1, MEDIA_ROWS, WIDTH, device=arrays["image_pixels"].device)
    targets = torch.zeros_like(features)
    features[:, :IMAGE_ROWS, 640:2176] = arrays["image_pixels"]
    features[:, IMAGE_ROWS + 1:, 576:640] = arrays["geometry_noisy"]
    targets[:, IMAGE_ROWS + 1:, 320:384] = arrays["geometry_canonical_velocity"]
    return {**batch, "native_media_features": features, "native_media_targets": targets}


def tensor_sha(value):
    import torch
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(A.canonical({"dtype": str(value.dtype), "shape": list(value.shape)}))
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validate_batch(batch, record, prefix, image, clean, sigma):
    import torch
    expected = build_batch(record, prefix, image, clean, sigma=sigma)
    if (set(batch) != set(expected) or any(v.requires_grad for v in batch.values()) or
        any(batch[k].dtype != v.dtype or batch[k].shape != v.shape or not torch.equal(batch[k].cpu(), v)
            for k, v in expected.items())):
        raise ValueError("native geometry batch differs from exact source/noise/chart/target reconstruction")


def velocity_loss(prediction, batch):
    """Existing raw head predicts clean-noise; canonical target is noise-clean."""
    import torch
    types, weights = batch["native_media_type_ids"], batch["native_media_loss_weights"]
    if (prediction.shape != (1, MEDIA_ROWS, WIDTH) or not prediction.is_floating_point() or
        not bool(torch.isfinite(prediction).all()) or types.shape != (1, MEDIA_ROWS) or
        not torch.equal(weights, batch["native_media_loss_mask"]) or
        not torch.equal(weights.cpu(), torch.cat((torch.zeros(1, IMAGE_ROWS + 1), torch.ones(1, LATENT_ROWS)), 1)) or
        bool((types[weights > 0] != 7).any())):
        raise ValueError("raw native prediction and exact latent-only objective weights required")
    owned = weights > 0
    delta = -prediction[..., 320:384][owned.to(prediction.device)].float() - batch["native_media_targets"][..., 320:384][owned].to(prediction.device).float()
    return delta.square().mean()


def model_inputs(batch):
    """Only genuine prompt, raw pixels, time and noisy latent enter native model."""
    keys = ("input_ids", "native_media_features", "native_media_indices", "native_media_type_ids",
            "native_media_valid_mask", "native_media_positions", "native_media_noise_levels")
    return {**{k: batch[k] for k in keys}, "native_media_only": True, "native_media_target_kind": "velocity",
            "charted_source_positions": {"hy3d": batch["charted_source_positions"]}}


def native_forward_loss(model, batch, compute):
    """Actual native API, without vocab logits or target/clean tensors as inputs.

    Callers own the pinned native model, optimizer, full-depth LoRA ownership,
    snapshot/resume and assigned runtime. This helper never launches training.
    """
    output = model(**model_inputs(batch))
    if not isinstance(output, dict) or set(output) != {"native_media_prediction"}:
        raise ValueError("actual native media-only prediction API required")
    return compute(output["native_media_prediction"])


def load_amendment(path, expected):
    if A.file_sha(path) != expected:
        raise ValueError("exact reviewed proposed geometry amendment file required")
    m = json_file(path)
    if (m.get("schema") != AMENDMENT_SCHEMA or m.get("policy") != POLICY or m.get("stage") != STAGE or
        m.get("objective") != OBJECTIVE or m.get("sigma_grid") != SIGMAS or
        m.get("prediction_kind") != "raw" or m.get("parent_signs") != {"7": -1} or
        m.get("target_kind") != "canonical_velocity" or m.get("native_parent_xyz_parity") is not False or
        m.get("full_trunk_parent_parity") is not False or m.get("guided_teacher") is not False or
        m.get("time_mapping") != "native_progress=1-canonical_sigma" or
        m.get("storage") != COMPACT or m.get("native_source_pins") != P.NATIVE_PINS or
        m.get("clone_config_sha256") != P.CONFIG_SHA or m.get("clone_index_sha256") != P.INDEX_SHA):
        raise ValueError("separate dated position-free/raw-sign/native-progress objective amendment required")
    return m


def inventory(intersection, expected):
    m = G.seal(intersection, expected, G.SCHEMA)
    if (m.get("sealed") is not True or m.get("complete_source_traversal") is not True or m.get("rows") != 24 or
        m.get("source_train_eligible") != 8 or m.get("held_sources_retained") != 16):
        raise ValueError("full actual24-source clean target/image/global intersection required")
    for ref in m["sources"].values():
        A.pinned(ref, "immutable geometry target inventory input")
    source_ref = m["sources"]["geometry_preparation"]
    _, streams = C.preparation(Path(source_ref["path"]), source_ref["sha256"])
    ref = m["outputs"]["prepared_targets"]
    values = list(A.rows(A.pinned(ref, "actual complete target intersection")))
    index = {r["id"]: r for r in values}
    if len(values) != 24 or len(index) != 24 or set(index) != set(streams["source_ready"]):
        raise ValueError("full source and target identity sets differ")
    for ident, row in index.items():
        source, admitted = streams["source_ready"][ident], streams["admission"][ident]
        if (row["source_row_sha256"] != A.digest(source) or row["source_admission_sha256"] != A.digest(admitted) or
            row["source_train_eligible"] != admitted["train_admitted"] or row["split"] != source["split"] or
            row["recovery_group_id"] != admitted["recovery_group_id"] or row.get("prepared_clean_target") is not True or
            row["target_only"]["encoder_query_xyz_generation_input_allowed"] is not False):
            raise ValueError("actual source/global eligibility/target/held join differs")
        for ref in list(row["inputs"].values()) + [v for v in row["target_only"].values() if isinstance(v, dict)]:
            A.pinned(ref, "actual source input or target-only artifact")
    return m, streams, index


def selection(path, amendment_sha, intersection_sha, index):
    """Root review selects exact source rows; preparation alone cannot approve."""
    s = json_file(path)
    if (s.get("schema") != SELECTION_SCHEMA or s.get("selected") is not True or
        s.get("amendment_sha256") != amendment_sha or s.get("intersection_sha256") != intersection_sha or
        s.get("geometry_quality_reviewed") is not True or not isinstance(s.get("review_basis"), str) or
        not s["review_basis"].strip() or s.get("no_held_sources_admitted") is not True):
        raise ValueError("source-specific reviewed quality and exact selected amendment required")
    approved = s.get("accepted_sources", [])
    ids = {r["id"] for r in approved}
    if not ids or len(ids) != len(approved):
        raise ValueError("unique actual quality-reviewed training sources required")
    for r in approved:
        row = index.get(r["id"], {})
        if (row.get("split") != "train" or row.get("source_train_eligible") is not True or
            r.get("source_row_sha256") != row.get("source_row_sha256") or
            r.get("native_receipt_sha256") != row.get("target_only", {}).get("native_receipt", {}).get("sha256")):
            raise ValueError("held, unavailable or changed native geometry source cannot be selected")
    return s, ids


def prepare(*, intersection, intersection_sha256, amendment, amendment_sha256, tokenizer_dir, output,
            selected=None, max_length=8192):
    if output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("fresh namespace and CUDA-hidden CPU-only geometry preparation required")
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    m = load_amendment(amendment, amendment_sha256)
    manifest, streams, index = inventory(intersection, intersection_sha256)
    if (m["evidence"]["target_intersection"]["sha256"] != intersection_sha256 or
        m["evidence"]["c11_parity"]["sha256"] != manifest["sources"]["original_C11_Torch_parity"]["sha256"]):
        raise ValueError("amendment must bind the exact observed target/image evidence")
    # Bind actual measured original bridge/time/decoder proof independently.
    proof_ref = m["evidence"]["bridge_time_decoder"]
    proof_path = A.pinned(proof_ref, "actual bridge/time/decoder proof")
    proof = G.seal(proof_path, proof_ref["sha256"], P.SCHEMA, "receipt_sha256")
    if (proof.get("sealed") is not True or proof.get("weights_unchanged") is not True or
        proof.get("trunk_executed") is not False or proof.get("native_source_pins") != P.NATIVE_PINS or
        proof.get("original_scheduler", {}).get("native_t0_is_noise_t1_is_data") is not True):
        raise ValueError("actual unmodified original native bridge/time evidence required")
    for r in proof["sources"]:
        if not r["decoder"]["physical_query_logit_invariance"]["allclose_atol2e-4_rtol2e-4"]:
            raise ValueError("original actual decoder permutation evidence required")
        for b in r["bridge"]:
            if not b["actual_noise_proj_input_exact_direct_progress_fourier"] or not b["positionfree_slot_permutation_effect"]["allclose_atol2e-4_rtol2e-4"]:
                raise ValueError("actual source-blind position-free/time boundary evidence required")
    allowed = {ident for ident, row in index.items() if row["source_train_eligible"] and row["split"] == "train"}
    selected_ref = None
    if selected is not None:
        _, allowed = selection(selected, amendment_sha256, intersection_sha256, index)
        selected_ref = pin(selected)
    for name, sha in TOKENIZER_PINS.items():
        if A.file_sha(tokenizer_dir / name) != sha:
            raise ValueError("genuine parent tokenizer file changed: " + name)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, local_files_only=True, trust_remote_code=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".geometry-flow-", dir=output.parent))
    entries, excluded = [], []
    for number, (ident, row) in enumerate(index.items()):
        record = streams["source_ready"][ident]
        prefix = pack_prefix(record, tokenizer, max_length=max_length)
        if ident not in allowed:
            excluded.append({"id": ident, "split": row["split"], "recovery_group_id": row["recovery_group_id"]})
            continue
        clean = torch.from_numpy(G.targets(Path(row["target_only"]["latent"]["path"]))["clean_model_space_latent"].copy())
        with np.load(row["inputs"]["raw_image_ingress"]["path"], allow_pickle=False) as loaded:
            image = {k: loaded[k].copy() for k in loaded.files}
        prefix_path, record_path = temporary / f"{number:03d}.prefix.json", temporary / f"{number:03d}.source.json"
        prefix_path.write_bytes(A.canonical(prefix) + b"\n")
        record_path.write_bytes(A.canonical(record) + b"\n")
        def final_pin(path):
            return {"path": str(output.resolve() / path.name), "sha256": A.file_sha(path)}
        for j, sigma in enumerate(SIGMAS):
            batch = build_batch(record, prefix, image, clean, sigma=sigma)
            stored = compact(batch)
            validate_batch(inflate(stored), record, prefix, image, clean, sigma)
            tensor_path = temporary / f"{number:03d}.sigma{j}.safetensors"
            save_file(stored, str(tensor_path))
            receipt = {"schema": "recovery_geometry_flow_cache.v1", "sealed": True, "stage": STAGE, "objective": OBJECTIVE,
                "id": ident + f":sigma{j}", "source_id": ident, "source_row_sha256": A.digest(record),
                "source_record": final_pin(record_path), "prefix": final_pin(prefix_path), "intersection": pin(intersection),
                "source_entry": row, "split": "train", "held": False, "recovery_group_id": row["recovery_group_id"],
                "sigma": sigma, "noise_seed": noise_seed(A.digest(record), sigma), "native_progress": 1 - sigma,
                "prediction_kind": "raw", "parent_signs": {"7": -1}, "target_kind": "canonical_velocity",
                "policy": POLICY, "storage": COMPACT, "amendment": pin(amendment), "selection": selected_ref,
                "tokenizer_files": {name: pin(tokenizer_dir / name) for name in TOKENIZER_PINS},
                "tensors": final_pin(tensor_path), "tensor_hashes": {k: tensor_sha(v) for k, v in stored.items()},
                "adapter_sha256": A.file_sha(Path(__file__)), "dependencies": dependency_pins(), "guided_teacher": False, "clean_target_xyz_in_inputs": False,
                "native_parent_xyz_parity": False, "full_trunk_parent_parity": False,
                "training_admission_selected": selected is not None, "training_started": False}
            receipt_path = temporary / f"{number:03d}.sigma{j}.receipt.json"
            receipt_path.write_bytes(A.canonical(receipt) + b"\n")
            entries.append({"id": receipt["id"], "source_id": ident, "tensors": final_pin(tensor_path),
                            "receipt": final_pin(receipt_path), "recovery_group_id": row["recovery_group_id"]})
    result = {"schema": SCHEMA, "sealed": True, "stage": STAGE, "objective": OBJECTIVE, "source_year": 2026,
        "intersection": pin(intersection), "amendment": pin(amendment), "selection": selected_ref,
        "complete_source_traversal": True, "heldout_groups_excluded": True, "batches": entries, "excluded": excluded,
        "sigma_grid": SIGMAS, "policy": POLICY, "storage": COMPACT, "adapter_sha256": A.file_sha(Path(__file__)), "dependencies": dependency_pins(),
        "training_admission_selected": selected is not None, "native_train_ready": False, "training_started": False,
        "blockers": ["Native production driver/insertion/objective and before/after held geometry proof remain required."] +
                    ([] if selected is not None else ["Exact owner-selected amendment and per-source quality review required."])}
    result["manifest_sha256"] = A.digest(result)
    (temporary / "manifest.json").write_bytes(A.canonical(result) + b"\n")
    temporary.rename(output)
    return result


class GeometryFlowAdapter:
    """Bind one authenticated supervised cache to a native prediction callback."""
    def __init__(self, receipt_path, receipt_sha256):
        self.receipt_path = Path(receipt_path)
        self.receipt = json_file(A.pinned({"path": str(receipt_path), "sha256": receipt_sha256}, "supervised geometry cache receipt"))
        r = self.receipt
        if (r.get("schema") != "recovery_geometry_flow_cache.v1" or r.get("sealed") is not True or
            r.get("stage") != STAGE or r.get("objective") != OBJECTIVE or r.get("policy") != POLICY or
            r.get("prediction_kind") != "raw" or r.get("parent_signs") != {"7": -1} or r.get("target_kind") != "canonical_velocity" or
            r.get("guided_teacher") is not False or r.get("split") != "train" or r.get("held") is not False or
            r.get("storage") != COMPACT or r.get("adapter_sha256") != A.file_sha(Path(__file__)) or r.get("dependencies") != dependency_pins() or
            r.get("training_admission_selected") is not True or not r.get("selection")):
            raise ValueError("separate selected source-sealed supervised geometry cache required")
        load_amendment(Path(r["amendment"]["path"]), r["amendment"]["sha256"])
        _, _, index = inventory(Path(r["intersection"]["path"]), r["intersection"]["sha256"])
        _, ids = selection(A.pinned(r["selection"], "actual geometry owner selection"), r["amendment"]["sha256"], r["intersection"]["sha256"], index)
        if r["source_id"] not in ids or r["source_entry"] != index[r["source_id"]]:
            raise ValueError("source cache is not in the exact selected native quality inventory")

    def bind(self, batch):
        import torch
        r = self.receipt
        record = json_file(A.pinned(r["source_record"], "actual authored volume source record"))
        prefix = json_file(A.pinned(r["prefix"], "actual parent generation prefix"))
        if A.digest(record) != r["source_row_sha256"] or r["source_id"] != record["id"]:
            raise ValueError("supervised geometry source row changed")
        if set(r["tokenizer_files"]) != set(TOKENIZER_PINS):
            raise ValueError("complete actual parent tokenizer pins required")
        for name, ref in r["tokenizer_files"].items():
            if ref["sha256"] != TOKENIZER_PINS[name]:
                raise ValueError("authentic parent tokenizer identity changed")
            A.pinned(ref, "actual parent tokenizer file")
        A.pinned(r["tensors"], "sealed original compact native cache")
        stored = compact(batch)
        if {k: tensor_sha(v) for k, v in stored.items()} != r["tensor_hashes"]:
            raise ValueError("provided native tensors differ from sealed actual supervised cache")
        row = r["source_entry"]
        clean = torch.from_numpy(G.targets(A.pinned(row["target_only"]["latent"], "actual clean VAE target"))["clean_model_space_latent"].copy())
        with np.load(A.pinned(row["inputs"]["raw_image_ingress"], "actual C11 image pixels"), allow_pickle=False) as loaded:
            image = {k: loaded[k].copy() for k in loaded.files}
        validate_batch(batch, record, prefix, image, clean, r["sigma"])
        return lambda prediction: velocity_loss(prediction, batch)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("intersection", "amendment", "tokenizer-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("intersection-sha256", "amendment-sha256"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--selected", type=Path)
    p.add_argument("--max-length", type=int, default=8192)
    result = prepare(**vars(p.parse_args(argv)))
    print(json.dumps({"batches": len(result["batches"]), "excluded": len(result["excluded"]),
                      "training_admission_selected": result["training_admission_selected"], "native_train_ready": False}))


if __name__ == "__main__":
    main()
