"""ingest.model_scan -- every weight directory on the AI server -> model (census in model.config).

Walks $OC, $L, /home/cereal/models and /home/cereal/model-surgery-lab (read-only; safetensors HEADERS
only, never tensor data).  A directory is a model artifact when it holds *.safetensors, or config.json
beside *.bin/*.pt; each GGUF file (multi-part files grouped) is its own artifact.

  * A directory that the ledger already registered keeps its ledger row (model_key = checkpoint id,
    role/stage/family from the ledger); the scan only adds size, dtype and the header census.
  * Everything else is keyed 'path:<sha256(host:path)[:16]>' with role/stage/family/project from the
    path rules below (ROLE_RULES; first match wins).
  * Not scanned as models: teacher_cache/ (datasets: ingest teacher_caches), pytest temp trees,
    venvs/site-packages/.git, and directories with more than BIG_DIR entries (shard dumps).
  * A model row this source wrote earlier whose path no longer exists is marked status 'missing'.
"""
from __future__ import annotations

import os
import re

from . import common as C
from .common import db

SOURCE = "model_scan"
BIG_DIR = 4000
MAX_HEADER_SHARDS = 120
SKIP_NAMES = {"__pycache__", ".git", "node_modules", "venv", ".venv", "site-packages", "teacher_cache"}
SKIP_PATH = re.compile(r"pytest-tmp|pytest-of-|/\.pytest_cache|/tmp[^/]*pytest")

# (regex on path, schema role, project, stage, family, scan label)
ROLE_RULES = [
    (r"/omni1/OC02-F-Q38-27BU-OMNI1$", "composite", "chimera-omni", "OMNI1", "Qwen3.8-27B-U", "omni1_composite"),
    (r"/clones/", "clone", "chimera-omni", None, "Qwen3.8-27B-U", "ledger_clone"),
    (r"/deltas/", "delta", "chimera-omni", "D1", "Qwen3.8-27B-U", "d1_delta"),
    (r"/distill/d1/", "intermediate", "chimera-omni", "D1", "Qwen3.8-27B-U", "d1_rung_or_run"),
    (r"/distill/fullkd/run1/out/student$", "student", "chimera-omni", "fullkd-run1", "Qwen3.8-27B-U", "fullkd_student"),
    (r"/distill/fullkd/", "intermediate", "chimera-omni", "fullkd", "Qwen3.8-27B-U", "fullkd_ckpt"),
    (r"/distill/media_m1/run1/out/export$", "media-unit", "media", "M1", "OMNI1-media", "media_m1_export"),
    (r"/distill/media_m1/", "intermediate", "media", "M1", "OMNI1-media", "media_m1_ckpt"),
    (r"/distill/m2/", "intermediate", "media", "M2", "OMNI1-media", "m2_artifact"),
    (r"/distill/d2/", "intermediate", "chimera-omni", "D2", "Qwen3.8-27B-U", "d2_ckpt"),
    (r"/qwen-fold/children", "fold-child", "neural-forge", "OC01", "Qwen3.5-4B", "fold_child_oc01_4b"),
    (r"/qwen-fold/dryrun", "intermediate", "neural-forge", "OC01-dryrun", "Qwen3.5-0.8B", "fold_dryrun_08b"),
    (r"/composite/s0/", "intermediate", "chimera-omni", "S0", "Qwen3.5-0.8B", "fold_s0_import_08b"),
    (r"/composite/c10/S5-", "fold-child", "chimera-omni", "S5", "Qwen3.8-27B-U", "fold_child_s5"),
    (r"/s3-fold-sep07-codex/S3-", "fold-child", "chimera-omni", "S3", "Qwen3.8-27B-U", "fold_child_s3"),
    (r"/s3-native-v2", "fold-child", "chimera-omni", "S3-native", "Qwen3.8-27B-U", "fold_child_s3_native"),
    (r"/research/fold/", "fold-child", "neural-forge", "fold-ladder", None, "fold_research_ladder"),
    (r"/composite/c10/design_sweep", "fold-child", "chimera-omni", "C10-sweep", "Qwen3.8-27B-U", "fold_c10_design_sweep"),
    (r"/composite/s7", "media-unit", "media", "S7", None, "fold_s7_media_insert"),
    (r"/runtime/", "intermediate", "media", "runtime-capture", None, "media_native_runtime_capture"),
    (r"/c17-smoke|/composite/c10/rehearsal|/composite/c17|/k3-c17-", "intermediate", "k3", "C17", "Kimi-K3", "fold_c17_k3_capture"),
    (r"/k3-continuation|/k3-diagnostic|/k3-fold-|/k3-nas|/omni-campaign/k3/|/k3-run-oct09", "intermediate", "k3", "K3-fold",
     "Kimi-K3", "fold_k3_intermediate"),
    (r"speech", "other", "speech", None, None, "speech_lane"),
    (r"/codex-", "other", "chimera-omni", None, None, "codex_lane_artifact"),
    (r"/ace-full28|/ace24|/ace/", "media-unit", "media", None, "ACE-Step", "media_ace_fold"),
    (r"/hy3d", "media-unit", "media", None, "Hunyuan3D-2.1", "media_hy3d_fold"),
    (r"/h3-|/h3/", "media-unit", "media", None, "MiniMax-H3", "media_h3_fold"),
    (r"/qimg", "media-unit", "media", None, "Qwen-Image", "media_qimg_fold"),
    (r"/native_executor", "other", "media", None, None, "native_executor"),
    (r"/large-fold-opt", "intermediate", "chimera-omni", None, None, "fold_large_opt"),
    (r"/export/", "export", "chimera-omni", None, None, "export"),
    (r"/import", "other", "chimera-omni", None, None, "import_lane"),
    (r"/parity/|/tps/", "intermediate", "chimera-omni", None, None, "parity_probe_state"),
    (r"/scratch|testwork|scratch_review_tests|/\.smoke|/\.review|/\.work", "intermediate", "chimera-omni", None, None,
     "scratch"),
    (r"^/home/cereal/model-surgery-lab/models", "other", "neural-forge", None, None, "nf_lab_model"),
    (r"^/home/cereal/model-surgery-lab", "intermediate", "neural-forge", None, None, "nf_lab_artifact"),
    (r"^/home/cereal/models/", "parent", "chimera-omni", "parent", None, "parent"),
]
FAMILY_HINTS = [(r"Qwen3\.8-27B|q38|Q38", "Qwen3.8-27B-U"), (r"Qwen3\.5-4B", "Qwen3.5-4B"),
                (r"Qwen3\.5-0\.8B|0\.8b|08b", "Qwen3.5-0.8B"), (r"LFM2", "LFM2.5-230M"), (r"Flash-Next|flash-next", "Qwen3.8-Flash-Next")]
GGUF_PART = re.compile(r"-\d{5}-of-\d{5}\.gguf$")


def classify(path: str) -> tuple[str, str, str | None, str | None, str]:
    rel = path
    for prefix, alias in ((C.OC, "/home/cereal/omni-campaign"), (C.LX, "/mnt/lexar/omni-campaign"),
                          (C.MODELS, "/home/cereal/models"), (C.NF_LAB, "/home/cereal/model-surgery-lab")):
        if path.startswith(prefix + "/") or path == prefix:
            rel = alias + path[len(prefix):]
            break
    for rx, role, project, stage, family, label in ROLE_RULES:
        if re.search(rx, rel):
            if family is None:
                family = next((f for frx, f in FAMILY_HINTS if re.search(frx, rel)), None)
            return role, project, stage, family, label
    return "other", "chimera-omni", None, next((f for frx, f in FAMILY_HINTS if re.search(frx, rel)), None), "other"


def config_essentials(directory: str) -> dict:
    cfg = C.load_json(os.path.join(directory, "config.json"), 8 << 20)
    if not isinstance(cfg, dict):
        return {}
    out: dict = {}
    if isinstance(cfg.get("config"), dict):            # chimera_omni.checkpoint.v1
        inner = cfg["config"]
        out["chimera_commit"] = cfg.get("chimera_commit")
        for k in ("architecture", "num_layers", "hidden_size", "intermediate_size", "vocab_size", "aux_heads",
                  "feed_forward_kind", "native_feature_dim", "native_type_count"):
            if k in inner:
                out[k] = inner[k]
        pattern = inner.get("block_pattern")
        if isinstance(pattern, list):
            out["block_pattern_counts"] = {t: pattern.count(t) for t in sorted(set(map(str, pattern)))}
        charted = inner.get("charted_blocks")
        if isinstance(charted, list):
            out["charted_blocks"] = [{"layer_index": b.get("layer_index"), "component": b.get("component")}
                                     for b in charted if isinstance(b, dict)]
        if isinstance(inner.get("ledger_bands"), (dict, list)):
            out["ledger_bands"] = inner["ledger_bands"]
        return out
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    for k in ("architectures", "model_type", "torch_dtype", "_class_name"):
        if k in cfg:
            out[k] = cfg[k]
    for k in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size",
              "vocab_size", "n_routed_experts", "num_experts", "num_experts_per_tok", "max_position_embeddings"):
        if k in text:
            out[k] = text[k]
    if isinstance(text.get("layer_types"), list):
        lt = [str(x) for x in text["layer_types"]]
        out["layer_types"] = {t: lt.count(t) for t in sorted(set(lt))}
    out["has_vision_config"] = "vision_config" in cfg
    out["has_audio_config"] = any("audio" in k for k in cfg)
    return out


def walk_models(roots: list[tuple[str, int]], errors: list[str]):
    """Yield (directory, files[DirEntry]) for every model directory under the roots."""
    seen = set()
    for root, maxdepth in roots:
        if not os.path.isdir(root):
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            if SKIP_PATH.search(d):
                continue
            try:
                entries = list(os.scandir(d))
            except OSError as exc:
                errors.append(f"scandir {d}: {exc.__class__.__name__}")
                continue
            files = [e for e in entries if e.is_file(follow_symlinks=False)]
            subs = [e for e in entries if e.is_dir(follow_symlinks=False) and e.name not in SKIP_NAMES]
            names = {e.name for e in files}
            is_model = (any(n.endswith((".safetensors", ".gguf")) for n in names)
                        or ("config.json" in names and any(n.endswith((".bin", ".pt", ".pth")) for n in names)))
            if len(entries) > BIG_DIR:
                errors.append(f"big dir not scanned ({len(entries)} entries): {d}")
                continue
            if is_model and d not in seen:
                seen.add(d)
                yield d, files
            if depth < maxdepth:
                stack.extend((os.path.join(d, s.name), depth + 1) for s in subs)


def _size(entries) -> int:
    total = 0
    for e in entries:
        try:
            total += e.stat(follow_symlinks=False).st_size
        except OSError:
            pass
    return total


def _write(conn, batch, host_id, path, files, kind_files) -> None:
    """Upsert one artifact (a directory, or one GGUF file group when kind_files is given)."""
    existing = C.model_id_by_path(conn, path)
    weights = kind_files if kind_files is not None else [
        e for e in files if e.name.endswith((".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt"))]
    size = _size(weights)
    if kind_files is None:
        shards = [e.path for e in files if e.name.endswith(".safetensors")]
        census = C.header_census(shards) if 0 < len(shards) <= MAX_HEADER_SHARDS else {}
        cfg = config_essentials(path)
        layout = ("chimera.checkpoint.v1" if "chimera_commit" in cfg or "block_pattern_counts" in cfg
                  else "hf" if cfg.get("architectures") or cfg.get("model_type") else "safetensors-raw")
        manifests = sorted(n for n in (e.name for e in files)
                           if n in ("composite_manifest.json", "manifest.json", "student_manifest.json", "receipt.json",
                                    "SHA256SUMS", "fold_receipt.json", "provenance.json", "export_receipt.json"))
    else:
        census, cfg, layout, manifests = {}, {}, "gguf", []
    config = dict(cfg)
    if census:
        config["census"] = {k: census[k] for k in ("params", "n_tensors", "dtype_params", "top_prefixes")}
    if manifests:
        config["manifests"] = manifests
    config["scan"] = {"n_files": len(files), "weight_files": len(weights), "mtime": C.iso_mtime(path)}
    dtype = C.main_dtype(census.get("dtype_params") or {}) if census else None
    params = census.get("params") or None
    if existing is not None:
        with conn.cursor() as cur:
            cur.execute("UPDATE model SET size_bytes = %s, dtype = COALESCE(%s, dtype), "
                        "parameter_count = COALESCE(parameter_count, %s), config = config || %s::jsonb, "
                        "status = 'present' WHERE id = %s",
                        (size, dtype, params, db.canonical_json({"scan": config}), existing))
        batch.wrote()
        return
    role, project, stage, family, label = classify(path)
    if layout == "gguf":
        role = "parent" if label == "parent" else "export"
    config["scan_label"] = label
    db.upsert_model(conn, C.sha16_path_key(path), project, role, layout, family=family, stage=stage, path=path,
                    host_id=host_id, parameter_count=params, size_bytes=size, dtype=dtype, config=config,
                    status="present", notes=f"scanned by {SOURCE} ({label})", prov=(path, None, C.SERVER_HOST),
                    batch_id=batch.id)
    batch.wrote()


def run(conn, batch: C.Batch, args) -> None:
    host_id = C.host_id(conn)
    roots = [(C.OC, 7), (C.LX, 6), (C.MODELS, 4), (C.NF_LAB, 4)]
    errors: list[str] = []
    seen_paths: set[str] = set()
    for directory, files in walk_models(roots, errors):
        ggufs = [e for e in files if e.name.endswith(".gguf")]
        non_gguf_weights = [e for e in files if e.name.endswith(".safetensors")]
        if ggufs:
            groups: dict[str, list] = {}
            for e in ggufs:
                groups.setdefault(GGUF_PART.sub(".gguf", e.path), []).append(e)
            for key_path, members in sorted(groups.items()):
                batch.saw()
                seen_paths.add(key_path)
                _write(conn, batch, host_id, key_path, files, members)
        if non_gguf_weights or not ggufs:
            batch.saw()
            seen_paths.add(directory)
            _write(conn, batch, host_id, directory, files, None)
    # artifacts this source wrote before that are gone now
    with conn.cursor() as cur:
        cur.execute("SELECT id, path FROM model WHERE notes LIKE %s AND status = 'present' AND ledger_checkpoint_id IS NULL",
                    (f"scanned by {SOURCE}%",))
        for mid, path in cur.fetchall():
            if path not in seen_paths and not os.path.exists(path):
                cur.execute("UPDATE model SET status = 'missing' WHERE id = %s", (mid,))
                batch.skip(f"now missing: {path}")
    for e in errors[:50]:
        batch.skip(e)
