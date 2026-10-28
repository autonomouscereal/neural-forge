#!/usr/bin/env python3
"""Tests for hy3d_voxel_set.py (raw 3D lane, 2026-10-22).

Synthetic meshes only: the pass-rule thresholds in RULES were set from these shapes before any Hunyuan3D object was
voxelised.  Needs numpy + scipy + Pillow (the AI server's /home/cereal/.venvs/hy3d-voxel).  The torch cross-check
against chimera_omni.native_media runs when NATIVE_MEDIA_PY names that module and torch imports; otherwise it skips.

    /home/cereal/.venvs/hy3d-voxel/bin/python -m unittest test_hy3d_voxel_set -v
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hy3d_voxel_set as hv  # noqa: E402


# ---------------------------------------------------------------------------------------------------------- meshes
def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return v, f


def merge(*parts):
    vs, fs, base = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + base)
        base += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def icosphere(subdiv=3, r=1.0):
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7],
         [9, 8, 1]]
    v = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v) * r, np.array(f, dtype=np.int64)


def torus(R=1.0, r=0.3, n=64, m=24):
    u, w = np.meshgrid(np.arange(n) * 2 * np.pi / n, np.arange(m) * 2 * np.pi / m, indexing="ij")
    v = np.stack(((R + r * np.cos(w)) * np.cos(u), r * np.sin(w), (R + r * np.cos(w)) * np.sin(u)), -1).reshape(-1, 3)
    f = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = i * m + j, ((i + 1) % n) * m + j, ((i + 1) % n) * m + (j + 1) % m, i * m + (j + 1) % m
            f += [[a, b, c], [a, c, d]]
    return v, np.array(f, dtype=np.int64)


def chair():
    """glTF Y-up: seat, back, four legs (each a closed box)."""
    parts = [box((-0.5, 0.45, -0.5), (0.5, 0.55, 0.5)), box((-0.45, 0.55, -0.5), (0.45, 1.4, -0.42))]
    for x in (-0.48, 0.42):
        for z in (-0.48, 0.42):
            parts.append(box((x, 0.0, z), (x + 0.06, 0.45, z + 0.06)))
    return merge(*parts)


def open_box():
    v, f = box((-1, -1, -1), (1, 1, 1))
    return v, f[[i for i in range(12) if i not in (6, 7)]]          # drop the +x face: an open container


def glb_bytes(v, f, node=None):
    pos = v.astype(np.float32).tobytes()
    idx = f.astype(np.uint32).reshape(-1).tobytes()
    binary = pos + idx
    binary += b"\x00" * ((-len(binary)) % 4)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [dict(node or {}, mesh=0)],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
           "buffers": [{"byteLength": len(binary)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
                           {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx)}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
                          "min": v.min(0).tolist(), "max": v.max(0).tolist()},
                         {"bufferView": 1, "componentType": 5125, "count": f.size, "type": "SCALAR"}]}
    js = json.dumps(doc).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)


def run_object(tmp, v, f):
    data = glb_bytes(v, f)
    sha = hashlib.sha256(data).hexdigest()
    p = Path(tmp, sha + ".glb")
    p.write_bytes(data)
    return hv.process_object({"mesh_sha256": sha, "mesh_store_path": str(p)})


class UnitTests(unittest.TestCase):
    def test_native_rows_roundtrip(self):
        raw = np.random.default_rng(0).uniform(-1, 1, (6, 64, 64, 64)).astype(np.float32)
        rows = hv.native_rows(raw)
        self.assertEqual(rows.shape, (512, 3072))
        self.assertTrue(np.array_equal(hv.rows_to_volume(rows, raw.shape), raw))
        # row 0 = the corner block, channel-major inside the row
        self.assertTrue(np.array_equal(rows[0][:512], raw[0, :8, :8, :8].reshape(-1)))
        self.assertTrue(np.array_equal(rows[1][:512], raw[0, :8, :8, 8:16].reshape(-1)))

    def test_native_rows_match_chimera_native_media(self):
        path = os.environ.get("NATIVE_MEDIA_PY", "")
        if not path:
            self.skipTest("NATIVE_MEDIA_PY not set")
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")
        spec = importlib.util.spec_from_file_location("native_media_under_test", path)
        nm = importlib.util.module_from_spec(spec)
        sys.modules["native_media_under_test"] = nm
        spec.loader.exec_module(nm)
        self.assertEqual(nm.V2_VOLUME_PATCH, hv.PATCH)
        self.assertEqual(nm.V2_RAW_UNIT_VALUES, hv.RAW_UNIT)
        self.assertEqual(tuple(nm.V2_VOLUME_CHANNELS), hv.CHANNELS)
        self.assertEqual(nm.V2_SIGNAL_RMS, hv.SIGNAL_RMS)
        sdf = np.random.default_rng(1).uniform(-20, 20, (64, 64, 64)).astype(np.float32)
        ours = hv.raw_volume(sdf)
        theirs = nm.sdf_volume_to_raw(torch.from_numpy(sdf)[None], truncation=hv.TRUNCATION_CELLS)[0].numpy()
        self.assertTrue(np.array_equal(ours, theirs))
        packet = nm.v2_volume_packet(torch.from_numpy(ours)[None])
        self.assertTrue(np.array_equal(packet.features[0].numpy(), hv.native_rows(ours)))
        back = nm.native_patches_to_volume(packet)[0].numpy()
        self.assertTrue(np.array_equal(back, ours))

    def test_glb_loader_applies_node_transforms(self):
        v, f = box((0, 0, 0), (1, 1, 1))
        lv, lf, facts = hv.load_glb(glb_bytes(v, f, {"translation": [1, 2, 3], "scale": [2, 2, 2]}))
        self.assertTrue(np.allclose(lv, v * 2 + [1, 2, 3]))
        self.assertTrue(np.array_equal(lf, f))
        self.assertEqual(facts["transformed_nodes"], 1)
        lv, _lf, facts = hv.load_glb(glb_bytes(v, f))
        self.assertTrue(np.allclose(lv, v))
        self.assertEqual(facts["transformed_nodes"], 0)
        with self.assertRaises(ValueError):
            hv.load_glb(b"not a glb at all, really not")

    def test_gltf_to_zup_is_a_rotation(self):
        self.assertTrue(np.allclose(hv.gltf_to_zup(np.array([[0.0, 1.0, 0.0]])), [[0, 0, 1]]))
        v, f = chair()
        self.assertAlmostEqual(hv.mesh_facts(v, f)["mesh_volume"], hv.mesh_facts(hv.gltf_to_zup(v), f)["mesh_volume"], 9)

    def test_mesh_facts(self):
        facts = hv.mesh_facts(*box((0, 0, 0), (1, 2, 3)))
        self.assertTrue(facts["watertight"])
        self.assertAlmostEqual(facts["mesh_volume"], 6.0, 9)
        self.assertAlmostEqual(facts["mesh_area"], 22.0, 9)
        self.assertFalse(hv.mesh_facts(*open_box())["watertight"])
        self.assertEqual(hv.mesh_facts(*chair())["components"], 6)

    def test_plan_candidates(self):
        def item(sha, status="admitted", split="train", text="a wooden chair"):
            return {"item_id": "i" + sha, "status": status, "split": split, "transcript": text, "asset_sha256": sha,
                    "store_path": f"/s/{sha}.glb", "bytes": 1, "batch_id": "3d/aiserve-mesh_hy3d-20261022",
                    "holdout_registry_version": 3, "width": None, "height": None, "reasons": [], "drop_path": None}

        def prov(sha, img, text="a wooden chair"):
            return {"artifact_sha256": sha, "source_artifact_sha256": img, "prompt": text,
                    "judge": {"rule": "mesh", "spec_sha256": "s" * 64}}
        meshes = [item("a"), item("b", split="train"), item("c", status="duplicate_near"), item("d"), item("e"),
                  item("f"), item("g")]
        images = {"ia": item("ia"), "ib": item("ib", status="held", split="test"), "ie": item("ie", status="review"),
                  "if": item("if")}
        provs = {"a": prov("a", "ia"), "b": prov("b", "ib"), "c": prov("c", "ia"), "d": None, "e": prov("e", "ie"),
                 "f": prov("f", "if", text="something else"), "g": prov("g", "missing")}
        jobs, excluded = hv.plan_candidates(meshes, images, provs)
        self.assertEqual([(j["mesh_sha256"], j["split"]) for j in jobs], [("a", "train"), ("b", "test")])
        why = {e["object_id"]: e["reason"].split(":")[0] for e in excluded}
        self.assertEqual(why, {"c": "intake_status", "d": "published_judged", "e": "intake_status",
                               "f": "prompt_matches_intake", "g": "intake_status"})
        self.assertEqual(hv.more_held("val", "train"), "val")
        self.assertEqual(hv.more_held("val", "test"), "test")


class VoxelTests(unittest.TestCase):
    """The shapes the RULES thresholds were set on.  Each prints its measured stats (the calibration record)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        print("\nCALIBRATION " + json.dumps({k: {"ok": r["ok"], "reason": r.get("reason"),
                                                  **{s: r["stats"][s] for s in ("occupancy_fraction", "surface_agreement",
                                                                                  "surface_abs_sdf_p99", "silhouette_iou",
                                                                                  "fill_gain", "volume_ratio")}}
                                              for k, r in cls.results.items()}, indent=1))
        cls.tmp.cleanup()

    def obj(self, name, v, f):
        r = run_object(self.tmp.name, v, f)
        self.results[name] = r
        return r

    def assert_passes(self, r):
        self.assertTrue(r["ok"], r.get("reason"))
        self.assertTrue(all(r["gates"].values()), r["gates"])

    def test_sphere(self):
        r = self.obj("sphere", *icosphere(4))
        self.assert_passes(r)
        raw = np.load(__import__("io").BytesIO(r["npz"]))["raw"].astype(np.float32)
        self.assertEqual(raw.shape, (6, 64, 64, 64))
        self.assertLess(raw[0, 32, 32, 32], -0.99)                       # deep inside: truncated at -1
        self.assertEqual(raw[0, 0, 0, 0], 1.0)                           # corner: outside
        self.assertTrue((raw[1:] == 0).all())                            # PBR absent -> 0
        analytic = 4 / 3 * np.pi * (hv.FIT * hv.GRID / 2) ** 3 / hv.GRID ** 3
        self.assertLess(abs(r["stats"]["occupancy_fraction"] - analytic) / analytic, 0.15)

    def test_box(self):
        self.assert_passes(self.obj("box", *box((-1, -0.5, -0.25), (1, 0.5, 0.25))))

    def test_torus(self):
        self.assert_passes(self.obj("torus", *torus()))

    def test_chair(self):
        r = self.obj("chair", *chair())
        self.assert_passes(r)
        # the back is at glTF -z = z-up +y: the occupied mass at high z (the back) sits at high y
        raw = np.load(__import__("io").BytesIO(r["npz"]))["raw"]
        occ = raw[0] < 0
        top = occ[48:]                                                    # z = 48..63: only the back is that high
        ys = np.nonzero(top.any(axis=(0, 2)))[0]
        self.assertGreater(ys.min(), 44)
        self.assertLess(ys.max(), 58)

    def test_thin_plate(self):
        self.assert_passes(self.obj("thin_plate", *box((-1, -0.01, -1), (1, 0.01, 1))))

    def test_open_box_is_refused_as_hollow(self):
        r = self.obj("open_box", *open_box())
        self.assertFalse(r["ok"])
        self.assertFalse(r["gates"]["not_hollow"])

    def test_silhouette_detects_a_mirrored_volume(self):
        v, f = chair()
        sha = "0" * 64
        rng = np.random.default_rng(3)
        vox = hv.voxelise(hv.gltf_to_zup(v), f, rng=rng)
        occ = vox["sdf_cells"] < 0
        pts = hv.sample_surface(vox["grid_vertices"], f, per_area=None, count=20000, rng=rng) / hv.POOL
        good = hv.silhouettes_iou(occ, pts)
        bad = hv.silhouettes_iou(occ[::-1], pts)                          # an upside-down volume
        self.assertGreaterEqual(min(good.values()), hv.RULES["per_object"]["silhouette_iou_min"])
        self.assertLess(min(bad.values()), hv.RULES["per_object"]["silhouette_iou_min"], (good, bad, sha))


class BuildTests(unittest.TestCase):
    def test_build_end_to_end_without_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, drop = Path(tmp, "store"), Path(tmp, "drop")
            store.mkdir()
            drop.mkdir()
            meshes, images = [], {}
            for i, (v, f) in enumerate((icosphere(3), chair(), box((-1, -1, -1), (1, 1, 1)))):
                data = glb_bytes(v, f)
                sha = hashlib.sha256(data).hexdigest()
                (store / f"{sha}.glb").write_bytes(data)
                img_sha = hashlib.sha256(f"img{i}".encode()).hexdigest()
                dp = drop / f"m{i}.glb"
                dp.write_bytes(data)
                Path(str(dp) + ".provenance.json").write_text(json.dumps({
                    "artifact_sha256": sha, "source_artifact_sha256": img_sha, "prompt": f"object {i}",
                    "judge": {"rule": "mesh", "spec_sha256": "c" * 64, "scores": {"caption": "x"}},
                    "params": {"seed": i, "source_prompt": f"object {i}, white"}, "generated_at": f"2026-10-22T0{i}:00:00Z",
                    "licence": "test", "synthetic": True}))
                meshes.append({"item_id": "m" * 63 + str(i), "status": "admitted", "split": "train", "transcript": f"object {i}",
                               "holdout_registry_version": 3, "reasons": [], "asset_sha256": sha,
                               "store_path": str(store / f"{sha}.glb"), "bytes": len(data),
                               "batch_id": "3d/aiserve-mesh_hy3d-20261022", "width": None, "height": None, "drop_path": str(dp)})
                images[img_sha] = {"item_id": "i" * 63 + str(i), "status": "held" if i == 1 else "admitted",
                                   "split": "val" if i == 1 else "train", "transcript": None, "holdout_registry_version": 3,
                                   "reasons": [], "asset_sha256": img_sha, "store_path": str(Path(tmp, "none.png")),
                                   "bytes": 1, "batch_id": "images/x", "width": 1024, "height": 1024, "drop_path": None}

            class FakeConn:
                def rollback(self):
                    pass

                def close(self):
                    pass
            old = (hv.corpus_connect, hv.fetch_items)
            hv.corpus_connect = lambda: FakeConn()
            hv.fetch_items = lambda conn, q, p: meshes if q is hv.MESH_QUERY else [images[s] for s in p if s in images]
            try:
                args = argparse.Namespace(version=1, root=str(Path(tmp, "set")), workers=2, limit=0, rules_run="test",
                                          no_omni_lab=True)
                self.assertEqual(hv.build(args), 0)
                with self.assertRaises(hv.Refusal):
                    hv.build(args)
            finally:
                hv.corpus_connect, hv.fetch_items = old
            d = Path(tmp, "set", "v1")
            self.assertTrue((d / "sealed").is_file())
            self.assertTrue(json.loads((d / "verify_receipt.json").read_text())["ok"])
            rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(sorted(r["split"] for r in rows), ["train", "train", "val"])
            manifest = json.loads((d / "manifest.json").read_text())
            self.assertEqual(manifest["items"], 3)
            self.assertEqual(manifest["rules_sha256"], hv.rules_sha256())
            self.assertEqual(manifest["splits"], {"rows_by_split": {"train": 2, "val": 1}})
            for r in rows:
                raw = np.load(d / r["volume"]["path"])["raw"]
                self.assertEqual(hashlib.sha256(raw.tobytes()).hexdigest(), r["volume"]["raw_sha256"])
                self.assertEqual(raw.dtype, np.float16)
            self.assertTrue((d / "previews" / "_sheet.png").is_file())
            # a tampered file fails verify
            victim = d / rows[0]["volume"]["path"]
            victim.write_bytes(victim.read_bytes() + b"x")
            self.assertFalse(hv.verify_dir(d)["ok"])


if __name__ == "__main__":
    unittest.main()
