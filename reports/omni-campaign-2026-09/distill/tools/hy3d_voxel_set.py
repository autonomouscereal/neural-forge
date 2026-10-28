#!/usr/bin/env python3
"""hy3d_voxel_set.py -- the raw 3D set: intake-registered Hunyuan3D 2.1 meshes -> v2 raw volumes, versioned.

Lane raw-3D data (Claude, 2026-10-22).  ARCHITECTURE_V2 section 4 fixes the raw 3D unit: an 8x8x8 voxel block x 6
channels (SDF, R, G, B, roughness, metallic) = 3,072 values in [-1, 1], cut by `native_media.v2_volume_packet`; a
64^3 object is 512 rows.  Section 14.4 preflight item 5 ("3D data. There is no raw voxel set") is what this closes.

Where the meshes come from (nothing here touches a GPU):
    gen-worker-media (GPU4, background tier, media-gate)  mesh_hy3d task = Qwen-Image 2.1 source image -> Hunyuan3D 2.1
      -> media_judge meaning check (VL on the source image + 4 grey renders, welded components, ground skirt)
      -> publish (judged-pass sets only) -> drop zone 3d/ + images/aiserve-mesh_hy3d-<date>/
      -> intake.py (cron): sha256, store copy, holdout screen of the prompt, grouping, split, near-dup refusal
      -> THIS TOOL: every mesh item the intake admitted or held, with its source image item, voxelised on CPU.
The canonical record stays the intake's (GLB + prompt + source image; R12 tokenizer-agnostic); a volume is a derived
view of it, like a teacher cache, and every row names the intake item ids it came from.

Recipe (the same steps as the round-4 proxy's ModelNet volumes, `proof/v2_proxy_data.mesh_to_sdf`, at 4x the grid):
    glTF Y-up -> z-up (x, -z, y); bounding-box centre, longest side fitted to 90 % of the box; surface sampled at
    256^3 (>= 4 samples per fine voxel face area, plus the vertices), one dilation, hole filling to a solid, exact
    Euclidean distance transform (outside minus inside), min-pooled over 4^3 blocks to 64^3 (a part one fine voxel
    thick survives), truncated at 8 cells = 1/8 of the box (the proxy's 2 cells at 16^3) and divided by it.
    Axes (z, y, x), z up.  Hunyuan3D 2.1 here is shape-only (the GPU4 graph has no paint stage), so colour,
    roughness and metallic are absent and are 0, exactly as `sdf_volume_to_raw` writes a missing channel.
    Stored as float16 [6, 64, 64, 64] in one compressed .npz per object.

Every object passes the pre-registered RULES (`rules` prints them and their sha256; the build records the sha) or
is listed under `excluded` with its reason.  Splits come from the intake: an object takes the most held split of its
mesh item and its source image item (an image->3D pair is one object; train < val < test).

Commands (CPU only; run under `nice -n 15`):
    rules                                     print the pass rules and their sha256
    build --version N [--workers 8] [--limit K] [--no-omni-lab]
                                              candidates from chimera_corpus -> $ROOT/v<N>/ (built in .v<N>.building,
                                              renamed when complete; an existing v<N> is never touched)
    verify --version N                        re-hash every file of v<N> against its SHA256SUMS

Output $ROOT/v<N>/ (ROOT default /mnt/lexar/omni-campaign/corpus/derived/hy3d_voxel64):
    rows.jsonl            one row per object (ids, texts, split, volume path + sha256, stats, gates, provenance)
    volumes/<id[:2]>/<id>.npz    raw float16 [6, 64, 64, 64]
    previews/<id>.png     source image + top / front / side depth views of the volume; previews/_sheet.png
    manifest.json         counts, splits, excluded, sizes, rates, dataset RMS, rules sha, tool sha, source query
    SHA256SUMS, verify_receipt.json, sealed

Credentials: the chimera_corpus role from the chimera-intake env file (vault key chimera_corpus_pg, env
CHIMERA_CORPUS_PG_PASSWORD), read-only session.  psycopg2, raw parameterised SQL only.
Exit codes: 0 ok, 2 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

TOOL = "hy3d_voxel_set"
VERSION = "1.0.0"   # 1.0.0 (2026-10-22): first version
SCHEMA = "omni-campaign.raw3d.hy3d-voxel-set.v1"

GRID = 64                       # 64^3 = 512 raw rows (ARCHITECTURE_V2 section 4 token table)
FINE = 256                      # surface sampling / fill / distance transform grid
POOL = FINE // GRID
FIT = 0.9                       # longest side of the bounding box = 90 % of the box (the proxy's rule)
TRUNCATION_CELLS = 8.0          # in 64^3 cells = 1/8 of the box = the proxy's 2 cells at 16^3
PATCH = 8                       # native_media.V2_VOLUME_PATCH
RAW_UNIT = 3072                 # native_media.V2_RAW_UNIT_VALUES
CHANNELS = ("sdf", "red", "green", "blue", "roughness", "metallic")   # native_media.V2_VOLUME_CHANNELS
SIGNAL_RMS = 0.5                # native_media.V2_SIGNAL_RMS (the loader applies ONE constant gain per dataset)
SPLIT_RANK = {"train": 0, "val": 1, "test": 2}
SAMPLE_SEED = 20261022
MAX_SURFACE_SAMPLES = 24_000_000   # per object (about 0.6 GB of float64 points at the cap)

DEFAULT_ROOT = "/mnt/lexar/omni-campaign/corpus/derived/hy3d_voxel64"
INTAKE_ENV = os.environ.get("HY3D_VOXEL_INTAKE_ENV", "/home/cereal/.config/chimera-intake/env")
OMNI_LAB = {"python": "/home/cereal/.venvs/chimera-intake/bin/python", "cli": "/home/cereal/omni-campaign/lab/omni_lab.py",
            "env_file": "/home/cereal/.config/omni-lab/env"}
MESH_BATCH_RE = r"^3d/aiserve-mesh_hy3d-[0-9]{8}$"
USABLE_STATUS = ("admitted", "held")

# Pre-registered pass rules.  Thresholds were set on synthetic shapes only (test_hy3d_voxel_set.py), before any
# Hunyuan3D object was voxelised; changing any value is a new rules_version and a new dataset version.
RULES: dict[str, Any] = {
    "rules_version": "1.0.0",
    "recipe": {"grid": GRID, "fine_grid": FINE, "pool": f"min over each {POOL}^3 block", "fit": FIT,
               "truncation_cells": TRUNCATION_CELLS, "axes": "(z, y, x), z up; glTF Y-up mapped (x, -z, y)",
               "channels": list(CHANNELS), "pbr": "absent in Hunyuan3D 2.1 shape-only output -> 0 (sdf_volume_to_raw)",
               "storage": "float16 [6, 64, 64, 64], np.savez_compressed key 'raw'", "patch": PATCH, "raw_unit": RAW_UNIT,
               "surface_samples": "4 per fine voxel of area, plus the vertices; total capped at "
                                  f"{MAX_SURFACE_SAMPLES} (never under 1 per fine voxel of area)", "sample_seed": SAMPLE_SEED},
    "per_object": {
        "integrity": "the store copy re-hashes to the intake asset sha256",
        "published_judged": "a drop-zone <file>.provenance.json with artifact_sha256 = the mesh, judge.rule = mesh and a "
                            "judge.spec_sha256 (media-publish moves only verdict = pass AND group_verdict = pass sets)",
        "intake_status": "mesh item and its source image item both admitted or held",
        "prompt_matches_intake": "provenance prompt == the intake item's transcript (the screened text)",
        "min_triangles": 4,
        "occupancy_fraction": [0.002, 0.85],
        "border_outside": "every voxel on the outer face of the grid has sdf > 0",
        "surface_agreement": {"points": 20000, "max_abs_sdf_cells": 2.5, "min_fraction": 0.95},
        "silhouette_iou_min": 0.80,
        "hollow": "refused when the mesh is not watertight and fill_gain < 1.5, or watertight and "
                  "occupied volume / mesh volume < 0.7",
        "raw_rows": "512 rows x 3072 values, finite, in [-1, 1]; the inverse cut is bit-identical",
        "float16_max_abs_error": 1e-3,
    },
    "split": "the most held of the mesh item's and the source image item's intake splits (train < val < test)",
    "dedup": "identical float16 volume bytes: the first object id (sorted) is kept",
}


class Refusal(RuntimeError):
    pass


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rules_sha256() -> str:
    return hashlib.sha256(canonical(RULES).encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def utc(epoch: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if epoch is None else epoch))


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ======================================================================================================================
# GLB (glTF 2.0 binary): every triangle primitive, with node transforms applied
# ======================================================================================================================


def _accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    a = doc["accessors"][index]
    if "bufferView" not in a:
        raise ValueError(f"accessor {index} has no bufferView (sparse accessors are not read)")
    view = doc["bufferViews"][a["bufferView"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
    item = np.dtype(dtype).itemsize * width
    stride = view.get("byteStride") or item
    base = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    count = int(a["count"])
    if count == 0:
        return np.zeros((0, width), dtype=dtype)
    need = base + (count - 1) * stride + item
    if need > len(binary):
        raise ValueError(f"accessor {index} reads past the binary chunk ({need} > {len(binary)})")
    raw = np.frombuffer(binary, dtype=np.uint8, count=(count - 1) * stride + item, offset=base)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, item), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(count, width)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T          # glTF is column-major
    t = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    s = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = t
    return m


def load_glb(data: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    """GLB bytes -> (vertices [V, 3] float64 in glTF axes, triangles [F, 3] int64, facts)."""

    if len(data) < 20:
        raise ValueError("too small for a GLB")
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2.0 binary")
    jlen, jtype = struct.unpack_from("<I4s", data, 12)
    if jtype != b"JSON":
        raise ValueError("first chunk is not JSON")
    doc = json.loads(data[20:20 + jlen].decode("utf-8"))
    off = 20 + jlen
    binary = b""
    if off + 8 <= len(data):
        blen, btype = struct.unpack_from("<I4s", data, off)
        if btype == b"BIN\x00":
            binary = data[off + 8: off + 8 + blen]

    def primitives(mesh_index: int, world: np.ndarray):
        for prim in doc["meshes"][mesh_index].get("primitives") or []:
            if prim.get("mode", 4) != 4:
                continue
            pos = _accessor(doc, binary, prim["attributes"]["POSITION"]).astype(np.float64)
            if prim.get("indices") is not None:
                idx = _accessor(doc, binary, prim["indices"]).reshape(-1).astype(np.int64)
            else:
                idx = np.arange(len(pos), dtype=np.int64)
            idx = idx[: len(idx) // 3 * 3].reshape(-1, 3)
            if len(idx) and (idx.min() < 0 or idx.max() >= len(pos)):
                raise ValueError("triangle index out of range")
            yield pos @ world[:3, :3].T + world[:3, 3], idx

    parts: list[tuple[np.ndarray, np.ndarray]] = []
    transformed = 0
    scenes = doc.get("scenes") or []
    if scenes and doc.get("nodes"):
        scene = scenes[int(doc.get("scene", 0))]
        stack = [(int(n), np.eye(4)) for n in scene.get("nodes") or []]
        seen = set()
        while stack:
            index, parent = stack.pop()
            if index in seen:
                raise ValueError("node graph has a cycle")
            seen.add(index)
            node = doc["nodes"][index]
            world = parent @ _node_matrix(node)
            if not np.allclose(world, np.eye(4)):
                transformed += 1
            if "mesh" in node:
                parts.extend(primitives(int(node["mesh"]), world))
            stack.extend((int(c), world) for c in node.get("children") or [])
    else:
        for m in range(len(doc.get("meshes") or [])):
            parts.extend(primitives(m, np.eye(4)))
    if not parts:
        raise ValueError("no triangle primitives")
    verts, faces, base = [], [], 0
    for pos, idx in parts:
        verts.append(pos)
        faces.append(idx + base)
        base += len(pos)
    v, f = np.concatenate(verts), np.concatenate(faces)
    if not np.isfinite(v).all():
        raise ValueError("non-finite vertex")
    return v, f, {"primitives": len(parts), "transformed_nodes": transformed, "materials": len(doc.get("materials") or []),
                  "textures": len(doc.get("textures") or [])}


# ======================================================================================================================
# Mesh facts
# ======================================================================================================================


def gltf_to_zup(v: np.ndarray) -> np.ndarray:
    """glTF Y-up (x, y, z) -> z-up (x, -z, y): a rotation, so handedness and volume sign are kept."""

    return np.stack((v[:, 0], -v[:, 2], v[:, 1]), axis=1)


def weld(v: np.ndarray, f: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = v.min(0), v.max(0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    q = np.round((v - lo) / (diag * tol)).astype(np.int64)
    _u, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return v[first], inv.reshape(-1)[f]


def mesh_facts(v: np.ndarray, f: np.ndarray) -> dict[str, Any]:
    """Watertightness (every welded edge used by exactly two faces), signed volume, area, components."""

    wv, wf = weld(v, f)
    keep = (wf[:, 0] != wf[:, 1]) & (wf[:, 1] != wf[:, 2]) & (wf[:, 0] != wf[:, 2])
    wf = wf[keep]
    edges = np.sort(np.concatenate((wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]])), axis=1)
    _e, counts = np.unique(edges, axis=0, return_counts=True)
    a, b, c = wv[wf[:, 0]], wv[wf[:, 1]], wv[wf[:, 2]]
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    area = float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(wv)
    graph = coo_matrix((np.ones(3 * len(wf), dtype=np.int8),
                        (np.concatenate([wf[:, 0], wf[:, 1], wf[:, 2]]), np.concatenate([wf[:, 1], wf[:, 2], wf[:, 0]]))),
                       shape=(n, n))
    ncomp, _label = connected_components(graph, directed=False)
    return {"welded_vertices": int(n), "faces": int(len(wf)), "degenerate_faces": int((~keep).sum()),
            "watertight": bool(len(counts) > 0 and (counts == 2).all()),
            "boundary_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum()),
            "mesh_volume": abs(volume), "mesh_area": area, "components": int(ncomp)}


def sample_surface(grid_v: np.ndarray, f: np.ndarray, *, per_area: float | None, count: int | None,
                   rng: np.random.Generator) -> np.ndarray:
    """Points on the triangles: `per_area` samples per unit area of every face (at least one), or `count` points
    drawn area-weighted."""

    a, b, c = grid_v[f[:, 0]], grid_v[f[:, 1]], grid_v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if per_area is not None:
        # no per-face cap (a decimated flat top can be one triangle of 20,000 fine voxels); the total is bounded by
        # MAX_SURFACE_SAMPLES, never below one sample per fine voxel of area
        density = float(per_area)
        planned = float(np.ceil(area * density).sum())
        if planned > MAX_SURFACE_SAMPLES:
            density = max(1.0, density * MAX_SURFACE_SAMPLES / planned)
        per_face = np.maximum(1, np.ceil(area * density)).astype(np.int64)
        face = np.repeat(np.arange(len(f)), per_face)
    else:
        total = float(area.sum())
        if total <= 0.0:
            raise ValueError("zero surface area")
        face = rng.choice(len(f), size=int(count), p=area / total)
    u, w = rng.random(face.shape[0]), rng.random(face.shape[0])
    flip = u + w > 1.0
    u[flip], w[flip] = 1.0 - u[flip], 1.0 - w[flip]
    return a[face] + u[:, None] * (b[face] - a[face]) + w[:, None] * (c[face] - a[face])


# ======================================================================================================================
# Voxelisation and the native cut
# ======================================================================================================================


def voxelise(v_zup: np.ndarray, f: np.ndarray, *, rng: np.random.Generator) -> dict[str, Any]:
    """z-up mesh -> 64^3 truncated SDF in coarse cells (negative inside) + the facts the rules need."""

    from scipy import ndimage

    lo, hi = v_zup.min(0), v_zup.max(0)
    centre, extent = (lo + hi) / 2.0, float((hi - lo).max())
    if extent <= 0.0 or len(f) == 0:
        raise ValueError("degenerate mesh")
    grid_v = (v_zup - centre) / extent * (FIT * FINE) + FINE / 2.0                 # fine continuous coordinates
    points = np.concatenate((sample_surface(grid_v, f, per_area=4.0, count=None, rng=rng), grid_v))
    cells = np.clip(np.floor(points).astype(np.int64), 0, FINE - 1)
    shell = np.zeros((FINE, FINE, FINE), dtype=bool)
    shell[cells[:, 2], cells[:, 1], cells[:, 0]] = True                              # (z, y, x)
    shell = ndimage.binary_dilation(shell, iterations=1)
    solid = ndimage.binary_fill_holes(shell)
    fine_sdf = ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)
    coarse = fine_sdf.reshape(GRID, POOL, GRID, POOL, GRID, POOL).min(axis=(1, 3, 5)) / POOL
    return {"sdf_cells": coarse.astype(np.float32), "grid_vertices": grid_v, "centre": centre, "extent": extent,
            "fill_gain": float(solid.sum() / max(1, int(shell.sum()))), "shell_voxels": int(shell.sum()),
            "surface_samples": int(len(points))}


def raw_volume(sdf_cells: np.ndarray) -> np.ndarray:
    """[64]^3 SDF in cells -> [6, 64, 64, 64] raw unit values (sdf_volume_to_raw: truncate, divide; absent PBR = 0)."""

    raw = np.zeros((len(CHANNELS),) + sdf_cells.shape, dtype=np.float32)
    raw[0] = np.clip(sdf_cells / TRUNCATION_CELLS, -1.0, 1.0)
    return raw


def native_rows(raw: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """[C, D, H, W] -> [rows, C * patch^3]: the numpy twin of native_media.volume_to_native_patches (batch 1)."""

    c, d, h, w = raw.shape
    if d % patch or h % patch or w % patch:
        raise ValueError("volume sides must be multiples of the patch")
    gd, gh, gw = d // patch, h // patch, w // patch
    return (raw.reshape(c, gd, patch, gh, patch, gw, patch).transpose(1, 3, 5, 0, 2, 4, 6)
            .reshape(gd * gh * gw, c * patch ** 3))


def rows_to_volume(rows: np.ndarray, shape: tuple[int, int, int, int], patch: int = PATCH) -> np.ndarray:
    """The inverse of native_rows (native_media.native_patches_to_volume, batch 1)."""

    c, d, h, w = shape
    gd, gh, gw = d // patch, h // patch, w // patch
    return (rows.reshape(gd, gh, gw, c, patch, patch, patch).transpose(3, 0, 4, 1, 5, 2, 6).reshape(c, d, h, w))


def trilinear(volume: np.ndarray, coords_zyx: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.map_coordinates(volume, coords_zyx.T, order=1, mode="nearest")


def silhouettes_iou(occ: np.ndarray, points_coarse: np.ndarray) -> dict[str, float]:
    """IoU of the volume's occupancy projected along z, y and x against fresh mesh surface samples projected the same
    way (a closed surface's projection covers its solid's)."""

    cells = np.clip(np.floor(points_coarse).astype(np.int64), 0, GRID - 1)            # (x, y, z)
    out = {}
    for name, axis, cols in (("top", 0, (1, 0)), ("front", 1, (2, 0)), ("side", 2, (2, 1))):
        vox = occ.any(axis=axis)
        mesh = np.zeros((GRID, GRID), dtype=bool)
        mesh[cells[:, cols[0]], cells[:, cols[1]]] = True
        union = int((vox | mesh).sum())
        out[name] = float((vox & mesh).sum() / union) if union else 1.0
    return out


def depth_views(occ: np.ndarray) -> list[np.ndarray]:
    """Top / front / side depth images (uint8, near = bright) of an occupancy grid (z, y, x)."""

    views = []
    for axis, flip in ((0, True), (1, False), (2, False)):
        o = np.flip(occ, axis=axis) if flip else occ
        hit = o.any(axis=axis)
        first = np.argmax(o, axis=axis)
        img = np.where(hit, 255 - (first * 200 // GRID), 0).astype(np.uint8)
        if axis in (1, 2):
            img = img[::-1]                                                          # z up in the picture
        views.append(img)
    return views


# ======================================================================================================================
# One object
# ======================================================================================================================


def process_object(job: dict[str, Any]) -> dict[str, Any]:
    """Pure CPU work for one candidate: read, re-hash, voxelise, gate, encode.  Never raises: a failure is a result."""

    out: dict[str, Any] = {"object_id": job["mesh_sha256"], "ok": False}
    t0 = time.time()
    try:
        data = Path(job["mesh_store_path"]).read_bytes()
        if sha256_bytes(data) != job["mesh_sha256"]:
            return {**out, "reason": "integrity: store copy does not re-hash to the asset sha256"}
        v, f, glb = load_glb(data)
        if len(f) < RULES["per_object"]["min_triangles"]:
            return {**out, "reason": f"min_triangles: {len(f)}"}
        facts = mesh_facts(v, f)
        v_zup = gltf_to_zup(v)
        rng = np.random.default_rng([SAMPLE_SEED, int(job["mesh_sha256"][:8], 16)])
        vox = voxelise(v_zup, f, rng=rng)
        sdf = vox["sdf_cells"]
        occ = sdf < 0.0
        occ_frac = float(occ.mean())
        cell = vox["extent"] / (FIT * GRID)                                          # mesh units per 64^3 cell
        occ_volume = float(occ.sum()) * cell ** 3
        lo_f, hi_f = RULES["per_object"]["occupancy_fraction"]
        border = np.concatenate([sdf[0].ravel(), sdf[-1].ravel(), sdf[:, 0].ravel(), sdf[:, -1].ravel(),
                                 sdf[:, :, 0].ravel(), sdf[:, :, -1].ravel()])
        check_rng = np.random.default_rng([SAMPLE_SEED + 1, int(job["mesh_sha256"][:8], 16)])
        sa = RULES["per_object"]["surface_agreement"]
        fresh = sample_surface(vox["grid_vertices"], f, per_area=None, count=sa["points"], rng=check_rng) / POOL
        at_surface = trilinear(sdf, np.stack((fresh[:, 2], fresh[:, 1], fresh[:, 0]), axis=1) - 0.5)
        agree = float((np.abs(at_surface) <= sa["max_abs_sdf_cells"]).mean())
        sil = silhouettes_iou(occ, fresh)
        raw = raw_volume(sdf)
        rows = native_rows(raw)
        back = rows_to_volume(rows, raw.shape)
        raw16 = raw.astype(np.float16)
        f16_err = float(np.abs(raw16.astype(np.float32) - raw).max())
        volume_ratio = occ_volume / facts["mesh_volume"] if facts["mesh_volume"] > 0 else None
        hollow = ((not facts["watertight"] and vox["fill_gain"] < 1.5)
                  or (facts["watertight"] and volume_ratio is not None and volume_ratio < 0.7))
        gates = {
            "occupancy_fraction": lo_f <= occ_frac <= hi_f,
            "border_outside": bool((border > 0).all()),
            "surface_agreement": agree >= sa["min_fraction"],
            "silhouette_iou": min(sil.values()) >= RULES["per_object"]["silhouette_iou_min"],
            "not_hollow": not hollow,
            "raw_rows": bool(rows.shape == ((GRID // PATCH) ** 3, RAW_UNIT) and np.isfinite(rows).all()
                             and rows.min() >= -1.0 and rows.max() <= 1.0 and np.array_equal(back, raw)),
            "float16": f16_err <= RULES["per_object"]["float16_max_abs_error"],
        }
        stats = {"occupancy_fraction": round(occ_frac, 6), "surface_agreement": round(agree, 5),
                 "surface_abs_sdf_p50": round(float(np.median(np.abs(at_surface))), 4),
                 "surface_abs_sdf_p99": round(float(np.percentile(np.abs(at_surface), 99)), 4),
                 "silhouette_iou": {k: round(x, 4) for k, x in sil.items()}, "fill_gain": round(vox["fill_gain"], 4),
                 "volume_ratio": None if volume_ratio is None else round(volume_ratio, 4),
                 "occupied_volume": occ_volume, "float16_max_abs_error": f16_err, "surface_samples": vox["surface_samples"],
                 "glb": glb, "mesh": facts, "cell_size": cell, "centre_zup": [float(x) for x in vox["centre"]],
                 "extent": vox["extent"]}
        buf = io.BytesIO()
        np.savez_compressed(buf, raw=raw16)
        failed = [k for k, ok in gates.items() if not ok]
        return {**out, "ok": not failed, "reason": ("gates: " + ",".join(failed)) if failed else None, "gates": gates,
                "stats": stats, "npz": buf.getvalue(), "raw_sha256": sha256_bytes(raw16.tobytes()),
                "views": [x.tobytes() for x in depth_views(occ)], "seconds": round(time.time() - t0, 3),
                "rms_sums": [float(np.square(raw16[c].astype(np.float64)).sum()) for c in range(len(CHANNELS))],
                "values_per_channel": int(raw16[0].size)}
    except Exception as exc:  # noqa: BLE001  (one bad mesh is a recorded exclusion, never an aborted build)
        return {**out, "reason": f"error: {type(exc).__name__}: {exc}"[:400], "seconds": round(time.time() - t0, 3)}


# ======================================================================================================================
# Candidates (chimera_corpus, read only) and provenance
# ======================================================================================================================


def intake_env(path: str = INTAKE_ENV) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:]
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[k.strip()] = val
    return out


def corpus_connect():
    import psycopg2

    env = {**intake_env(), **{k: v for k, v in os.environ.items() if k.startswith("CHIMERA_CORPUS_PG_")}}
    password = env.get("CHIMERA_CORPUS_PG_PASSWORD", "")
    if not password:
        raise Refusal(f"CHIMERA_CORPUS_PG_PASSWORD is not set (vault key chimera_corpus_pg; env file {INTAKE_ENV})")
    return psycopg2.connect(host=env.get("CHIMERA_CORPUS_PG_HOST", "127.0.0.1"), port=int(env.get("CHIMERA_CORPUS_PG_PORT", "25490")),
                            dbname=env.get("CHIMERA_CORPUS_PG_DB", "chimera_corpus"), user=env.get("CHIMERA_CORPUS_PG_USER", "chimera_corpus"),
                            password=password, connect_timeout=10, application_name=f"{TOOL}-{VERSION}",
                            options="-c default_transaction_read_only=on -c statement_timeout=60000")


ITEM_COLUMNS = ("item_id", "status", "split", "transcript", "holdout_registry_version", "reasons", "asset_sha256",
                "store_path", "bytes", "batch_id", "width", "height", "drop_path")
MESH_QUERY = (
    "SELECT i.item_id, i.status, i.split, i.transcript, i.holdout_registry_version, i.reasons, a.sha256, a.store_path,"
    " a.bytes, a.batch_id, a.width, a.height,"
    " (SELECT f.drop_path FROM intake_file f WHERE f.sha256 = a.sha256 ORDER BY f.id LIMIT 1)"
    " FROM item i JOIN asset a ON a.sha256 = i.asset_sha256"
    " WHERE a.modality = 'mesh' AND a.batch_id ~ %s ORDER BY a.sha256")
IMAGE_QUERY = (
    "SELECT i.item_id, i.status, i.split, i.transcript, i.holdout_registry_version, i.reasons, a.sha256, a.store_path,"
    " a.bytes, a.batch_id, a.width, a.height, NULL"
    " FROM item i JOIN asset a ON a.sha256 = i.asset_sha256"
    " WHERE a.modality = 'image' AND a.sha256 = ANY(%s) ORDER BY a.sha256")


def fetch_items(conn, query: str, param: Any) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(query, (param,))
        rows = [dict(zip(ITEM_COLUMNS, r)) for r in cur.fetchall()]
    for r in rows:
        r["asset_sha256"] = r["asset_sha256"].strip()
        r["item_id"] = r["item_id"].strip()
    return rows


def more_held(a: str | None, b: str | None) -> str | None:
    if a is None or b is None:
        return None
    return a if SPLIT_RANK[a] >= SPLIT_RANK[b] else b


def plan_candidates(meshes: list[dict[str, Any]], images: dict[str, dict[str, Any]],
                    provenance: dict[str, dict[str, Any] | None]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pure: mesh items + their image items + provenance docs -> (jobs, excluded).  One object per mesh asset."""

    jobs, excluded = [], []
    by_asset: dict[str, list[dict[str, Any]]] = {}
    for m in meshes:
        by_asset.setdefault(m["asset_sha256"], []).append(m)
    for sha in sorted(by_asset):
        items = by_asset[sha]
        usable = [m for m in items if m["status"] in USABLE_STATUS]
        if not usable:
            excluded.append({"object_id": sha, "reason": f"intake_status: mesh item {items[0]['status']}"})
            continue
        m = usable[0]
        prov = provenance.get(sha)
        if not prov:
            excluded.append({"object_id": sha, "reason": "published_judged: no provenance beside the dropped file"})
            continue
        judge = prov.get("judge") or {}
        if prov.get("artifact_sha256") != sha or judge.get("rule") != "mesh" or not judge.get("spec_sha256"):
            excluded.append({"object_id": sha, "reason": "published_judged: provenance does not name this mesh as judged"})
            continue
        if (prov.get("prompt") or "").strip() != (m.get("transcript") or "").strip():
            excluded.append({"object_id": sha, "reason": "prompt_matches_intake: provenance prompt != intake transcript"})
            continue
        src_sha = prov.get("source_artifact_sha256")
        img = images.get(src_sha or "")
        if img is None:
            excluded.append({"object_id": sha, "reason": "intake_status: source image not registered by the intake (yet)"})
            continue
        if img["status"] not in USABLE_STATUS:
            excluded.append({"object_id": sha, "reason": f"intake_status: source image item {img['status']}"})
            continue
        split = more_held(m["split"], img["split"])
        if split is None:
            excluded.append({"object_id": sha, "reason": "split: an item has no split"})
            continue
        jobs.append({"mesh_sha256": sha, "mesh_store_path": m["store_path"], "mesh": m, "image": img, "provenance": prov,
                     "split": split})
    return jobs, excluded


def read_provenance(drop_path: str | None, sha: str) -> dict[str, Any] | None:
    if not drop_path:
        return None
    p = Path(drop_path + ".provenance.json")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("artifact_sha256") == sha else None


# ======================================================================================================================
# Build
# ======================================================================================================================


def preview_png(views: list[bytes], image_path: str | None) -> bytes:
    from PIL import Image

    tile = 128
    sheet = Image.new("L", (tile * 4, tile), 0)
    if image_path:
        try:
            with Image.open(image_path) as im:
                sheet.paste(im.convert("L").resize((tile, tile)), (0, 0))
        except OSError:
            pass
    for i, raw in enumerate(views):
        img = Image.frombytes("L", (GRID, GRID), raw).resize((tile, tile), Image.NEAREST)
        sheet.paste(img, (tile * (i + 1), 0))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def contact_sheet(pngs: list[bytes], columns: int = 4) -> bytes:
    from PIL import Image

    if not pngs:
        return b""
    tiles = [Image.open(io.BytesIO(p)) for p in pngs]
    w, h = tiles[0].size
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("L", (w * columns, h * rows), 40)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % columns) * w, (i // columns) * h))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def row_for(job: dict[str, Any], res: dict[str, Any], volume_rel: str, volume_sha: str) -> dict[str, Any]:
    prov, m, img = job["provenance"], job["mesh"], job["image"]
    judge = prov.get("judge") or {}
    params = prov.get("params") or {}
    return {
        "object_id": job["mesh_sha256"], "split": job["split"],
        "split_from": {"mesh": m["split"], "image": img["split"]},
        "text": {"prompt": m["transcript"], "source_prompt": params.get("source_prompt")},
        "volume": {"path": volume_rel, "sha256": volume_sha, "raw_sha256": res["raw_sha256"], "key": "raw",
                   "dtype": "float16", "shape": [len(CHANNELS), GRID, GRID, GRID], "channels": list(CHANNELS),
                   "rows": (GRID // PATCH) ** 3, "raw_unit": RAW_UNIT, "patch": PATCH,
                   "truncation_cells": TRUNCATION_CELLS, "fit": FIT, "axes": "(z, y, x), z up",
                   "cell_size": res["stats"]["cell_size"], "centre_zup": res["stats"]["centre_zup"],
                   "pbr": "absent (0)"},
        "mesh": {"item_id": m["item_id"], "asset_sha256": m["asset_sha256"], "store_path": m["store_path"],
                 "bytes": m["bytes"], "batch_id": m["batch_id"], "intake_status": m["status"],
                 "holdout_registry_version": m["holdout_registry_version"]},
        "image": {"item_id": img["item_id"], "asset_sha256": img["asset_sha256"], "store_path": img["store_path"],
                  "width": img["width"], "height": img["height"], "intake_status": img["status"],
                  "role": "the Qwen-Image 2.1 picture Hunyuan3D 2.1 was conditioned on"},
        "stats": {k: v for k, v in res["stats"].items() if k not in ("cell_size", "centre_zup")},
        "gates": res["gates"], "voxelise_seconds": res["seconds"],
        "provenance": {"gen_task_id": prov.get("gen_task_id"), "generated_at": prov.get("generated_at"),
                       "generator_tool": prov.get("tool"), "graph_sha256": prov.get("graph_sha256"),
                       "seed": params.get("seed"), "prompt_source": prov.get("prompt_source"),
                       "licence": prov.get("licence"), "synthetic": prov.get("synthetic") is True,
                       "judge": {"tool": judge.get("tool"), "spec_sha256": judge.get("spec_sha256"),
                                 "model_sha256": judge.get("model_sha256"), "omni_lab_run_id": judge.get("omni_lab_run_id"),
                                 "caption_unscreened": (judge.get("scores") or {}).get("caption")}},
    }


def omni_lab_log(*args: str) -> str | None:
    env = dict(os.environ)
    try:
        for line in Path(OMNI_LAB["env_file"]).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, val = line.split("=", 1)
                env.setdefault(k.strip(), val.strip().strip("'\""))
    except OSError:
        pass
    try:
        proc = subprocess.run([OMNI_LAB["python"], OMNI_LAB["cli"], "log-run", *args], env=env, capture_output=True,
                              text=True, timeout=60, cwd=str(Path(OMNI_LAB["cli"]).parent))
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[{TOOL}] omni_lab log-run failed: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"[{TOOL}] omni_lab log-run exit {proc.returncode}: {proc.stderr[-300:]}", file=sys.stderr)
        return None
    lines = proc.stdout.strip().splitlines()
    return lines[-1] if lines else None


def build(args) -> int:
    root = Path(args.root)
    final = root / f"v{args.version}"
    if final.exists():
        raise Refusal(f"{final} exists: a version is never rewritten")
    work = root / f".v{args.version}.building"
    if work.exists():
        stale = root / "_failed_builds" / f"v{args.version}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        stale.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work, stale)                                                      # never deleted
        print(f"[{TOOL}] moved a stale partial build to {stale}")
    started = time.time()
    tool_sha = sha256_file(__file__)
    rid = None
    if not args.no_omni_lab:
        rid = omni_lab_log("start", "--kind", "build", "--name", f"hy3d voxel set v{args.version}", "--owner", "claude",
                           "--config", canonical({"tool": f"{TOOL} {VERSION}", "tool_sha256": tool_sha,
                                                  "rules_sha256": rules_sha256(), "out": str(final),
                                                  "rules_review_run": args.rules_run}))
    conn = corpus_connect()
    try:
        meshes = fetch_items(conn, MESH_QUERY, MESH_BATCH_RE)
        provenance = {m["asset_sha256"]: read_provenance(m["drop_path"], m["asset_sha256"]) for m in meshes}
        src = sorted({p["source_artifact_sha256"] for p in provenance.values() if p and p.get("source_artifact_sha256")})
        images = {r["asset_sha256"]: r for r in fetch_items(conn, IMAGE_QUERY, src)}
        conn.rollback()
    finally:
        conn.close()
    jobs, excluded = plan_candidates(meshes, images, provenance)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[{TOOL}] {len(meshes)} mesh items, {len(jobs)} candidates, {len(excluded)} excluded before voxelising")
    (work / "volumes").mkdir(parents=True)
    (work / "previews").mkdir()
    rows, sheet_pngs, seen_raw = [], [], {}
    rms_sums = np.zeros(len(CHANNELS))
    values = 0
    seconds = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=os.nice, initargs=(15,)) as pool:
        for job, res in zip(jobs, pool.map(process_object, jobs, chunksize=1)):
            if "seconds" in res:
                seconds.append(res["seconds"])
            if not res["ok"]:
                excluded.append({"object_id": job["mesh_sha256"], "reason": res["reason"], "gates": res.get("gates"),
                                 "stats": {k: v for k, v in (res.get("stats") or {}).items() if k not in ("glb", "mesh")}})
                continue
            if res["raw_sha256"] in seen_raw:
                excluded.append({"object_id": job["mesh_sha256"], "reason": f"dedup: same volume as {seen_raw[res['raw_sha256']]}"})
                continue
            seen_raw[res["raw_sha256"]] = job["mesh_sha256"]
            oid = job["mesh_sha256"]
            rel = f"volumes/{oid[:2]}/{oid}.npz"
            (work / rel).parent.mkdir(exist_ok=True)
            (work / rel).write_bytes(res["npz"])
            png = preview_png(res["views"], job["image"]["store_path"])
            (work / "previews" / f"{oid}.png").write_bytes(png)
            if len(sheet_pngs) < 48:
                sheet_pngs.append(png)
            rows.append(row_for(job, res, rel, sha256_bytes(res["npz"])))
            if job["split"] == "train":
                rms_sums += np.asarray(res["rms_sums"])
                values += res["values_per_channel"]
    rows.sort(key=lambda r: r["object_id"])
    if sheet_pngs:
        (work / "previews" / "_sheet.png").write_bytes(contact_sheet(sheet_pngs))
    with open(work / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(canonical(r) + "\n")
    splits: dict[str, int] = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    reasons: dict[str, int] = {}
    for e in excluded:
        key = e["reason"].split(":")[0] if not e["reason"].startswith("gates") else e["reason"]
        reasons[key] = reasons.get(key, 0) + 1
    per_channel = (np.sqrt(rms_sums / values) if values else np.zeros(len(CHANNELS))).tolist()
    overall = float(math.sqrt(rms_sums.sum() / (values * len(CHANNELS)))) if values else None
    vol_bytes = [(work / r["volume"]["path"]).stat().st_size for r in rows]
    gen_times = sorted(r["provenance"]["generated_at"] for r in rows if r["provenance"]["generated_at"])
    manifest = {
        "schema": SCHEMA, "set": "hy3d_voxel64", "version": f"v{args.version}", "items": len(rows),
        "splits": {"rows_by_split": splits}, "excluded_by_reason": reasons, "excluded": excluded,
        "tool": f"{TOOL} {VERSION}", "tool_sha256": tool_sha, "rules": RULES, "rules_sha256": rules_sha256(),
        "rules_review_run": args.rules_run, "omni_lab_build_run": rid,
        "source": {"database": "chimera_corpus (intake.py registry), read-only", "mesh_query": MESH_QUERY,
                   "mesh_batch_pattern": MESH_BATCH_RE, "image_query": IMAGE_QUERY, "mesh_items_seen": len(meshes),
                   "usable_status": list(USABLE_STATUS)},
        "unit": {"raw_unit_values": RAW_UNIT, "rows_per_object": (GRID // PATCH) ** 3, "channels": list(CHANNELS),
                 "cut": "native_media.v2_volume_packet (8^3 x 6)", "range": [-1.0, 1.0]},
        "signal": {"train_rms_overall": overall, "train_rms_per_channel": dict(zip(CHANNELS, per_channel)),
                   "v2_signal_rms": SIGNAL_RMS,
                   "dataset_gain_to_v2_signal_rms": (SIGNAL_RMS / overall) if overall else None,
                   "note": "native_media.signal_gain(train_rms_overall): the loader applies this ONE gain to every row"},
        "sizes": {"volume_bytes_total": int(sum(vol_bytes)), "volume_bytes_mean": (sum(vol_bytes) / len(vol_bytes)) if vol_bytes else None,
                  "volume_bytes_max": max(vol_bytes) if vol_bytes else None,
                  "mesh_bytes_total": int(sum(r["mesh"]["bytes"] for r in rows)),
                  "raw_values_per_object": len(CHANNELS) * GRID ** 3},
        "rates": {"voxelise_seconds_mean": (sum(seconds) / len(seconds)) if seconds else None,
                  "voxelise_seconds_max": max(seconds) if seconds else None, "workers": args.workers,
                  "build_wall_seconds": round(time.time() - started, 1),
                  "generated_first": gen_times[0] if gen_times else None, "generated_last": gen_times[-1] if gen_times else None},
        "licence": "Hunyuan3D 2.1 output: TENCENT HUNYUAN NON-COMMERCIAL / community licence, territory limits apply; "
                   "prompts per row (Strata / COCO 2017 captions CC BY 4.0)",
        "built_at_utc": utc(), "host": os.uname().nodename if hasattr(os, "uname") else None,
    }
    write_atomic(work / "manifest.json", (json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode())
    sums = []
    for p in sorted(work.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            sums.append(f"{sha256_file(p)}  {p.relative_to(work).as_posix()}")
    write_atomic(work / "SHA256SUMS", ("\n".join(sums) + "\n").encode())
    os.replace(work, final)
    receipt = verify_dir(final)
    write_atomic(final / "verify_receipt.json", (json.dumps(receipt, indent=1) + "\n").encode())
    if receipt["ok"]:
        (final / "sealed").write_text(utc() + "\n", encoding="utf-8")
    if rid:
        metrics = [f"items={len(rows)}", f"excluded={len(excluded)}", f"volume_bytes_total={sum(vol_bytes)}"]
        metrics += [f"split_{k}={v}" for k, v in splits.items()]
        if seconds:
            metrics.append(f"voxelise_seconds_mean={sum(seconds) / len(seconds):.3f}")
        argv = ["end", rid, "--status", "pass" if receipt["ok"] and rows else "fail", "--exit-code", "0",
                "--receipt", str(final / "manifest.json")]
        for mtr in metrics:
            argv += ["--metric", mtr]
        omni_lab_log(*argv)
    print(json.dumps({"version": str(final), "items": len(rows), "splits": splits, "excluded_by_reason": reasons,
                      "volume_bytes_total": sum(vol_bytes), "voxelise_seconds_mean": manifest["rates"]["voxelise_seconds_mean"],
                      "build_wall_seconds": manifest["rates"]["build_wall_seconds"], "train_rms_overall": overall,
                      "verify_ok": receipt["ok"], "omni_lab_build_run": rid}, indent=1))
    return 0 if receipt["ok"] else 2


def verify_dir(d: Path) -> dict[str, Any]:
    bad, n = [], 0
    for line in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, rel = line.split("  ", 1)
        n += 1
        p = d / rel
        if not p.is_file() or sha256_file(p) != sha:
            bad.append(rel)
    listed = {line.split("  ", 1)[1] for line in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if line.strip()}
    extra = sorted(p.relative_to(d).as_posix() for p in d.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "verify_receipt.json", "sealed")
                   and p.relative_to(d).as_posix() not in listed)
    manifest_sha = sha256_file(d / "manifest.json") if (d / "manifest.json").is_file() else None
    return {"ok": not bad and not extra and n > 0, "files": n, "mismatched": bad, "unlisted": extra,
            "manifest_sha256": manifest_sha, "verified_utc": utc(), "tool": f"{TOOL} {VERSION}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rules")
    b = sub.add_parser("build")
    b.add_argument("--version", type=int, required=True)
    b.add_argument("--root", default=DEFAULT_ROOT)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--rules-run", default=None, help="omni_lab review run id that pre-registered RULES")
    b.add_argument("--no-omni-lab", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--version", type=int, required=True)
    v.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "rules":
            print(json.dumps({"rules": RULES, "rules_sha256": rules_sha256(), "tool": f"{TOOL} {VERSION}"}, indent=1))
            return 0
        if args.cmd == "build":
            return build(args)
        receipt = verify_dir(Path(args.root) / f"v{args.version}")
        print(json.dumps(receipt, indent=1))
        return 0 if receipt["ok"] else 2
    except Refusal as exc:
        print(f"[{TOOL}] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
