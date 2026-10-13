"""Verify collected native ShapeVAE targets and source-bound raw image ingress.

This creates a preparation index, not training admission. Clean encoder query XYZ
and physical oracles stay in target references. No model, GPU or teacher runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

try:
    from . import geometry_conditioning as C
    from . import hy3d_target_session as S
except ImportError:
    import geometry_conditioning as C
    import hy3d_target_session as S
A, G, H = S.A, S.G, S.H
SCHEMA = "recovery_geometry_target_intersection.v1"
TENSORS = {
    "clean_raw_latent": ("F32", [1, 4096, 64]),
    "clean_model_space_latent": ("F32", [1, 4096, 64]),
    "target_only_encoder_query_xyz": ("F32", [1, 4096, 3]),
    "physical_oracle_queries": ("F32", [1, 125, 3]),
    "native_geometry_logits": ("F32", [1, 125, 1]),
    "physical_oracle_inside": ("BOOL", [125]),
}


def seal(path, expected, schema, field="manifest_sha256"):
    path = Path(path)
    if not A.SHA.fullmatch(str(expected)) or A.file_sha(path) != expected:
        raise ValueError("immutable preparation/receipt file seal differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema or A.digest({k: v for k, v in value.items() if k != field}) != value.get(field):
        raise ValueError("schema/canonical preparation seal differs")
    return value


def collected_path(directory, spec):
    # Container and Windows paths differ; exact known basenames plus file seals
    # are preserved. Never follow a path obtained from teacher output.
    name = str(spec["path"]).replace("\\", "/").rsplit("/", 1)[-1]
    if name in ("", ".", ".."):
        raise ValueError("bounded collected basename required")
    path = directory / name
    if A.file_sha(path) != spec["sha256"]:
        raise ValueError("collected native artifact file seal differs")
    return path


def targets(path):
    """Read only the six exact finite F32/BOOL native target tensors.

    This deliberately small offline format reader avoids a Torch installation.
    Shapes, contiguous byte coverage, key set and dtype are strict; these are
    the genuine emitted safetensors files, not reconstructed model outputs.
    """
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError("native target safetensors header missing")
    length = struct.unpack("<Q", data[:8])[0]
    if not 2 <= length <= 16384 or length + 8 > len(data):
        raise ValueError("bounded native target header required")
    header = json.loads(data[8:8 + length])
    if set(header) != set(TENSORS):
        raise ValueError("only exact target-side native tensor keys allowed")
    start = 8 + length
    result, spans = {}, []
    for name, (dtype, shape) in TENSORS.items():
        spec = header[name]
        if set(spec) != {"dtype", "shape", "data_offsets"} or spec["dtype"] != dtype or spec["shape"] != shape:
            raise ValueError("native target tensor dtype/shape differs")
        offsets = spec["data_offsets"]
        if len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError("native target byte offsets invalid")
        lo, hi = offsets
        width = 4 if dtype == "F32" else 1
        if lo < 0 or hi - lo != int(np.prod(shape)) * width or start + hi > len(data):
            raise ValueError("native target byte coverage differs")
        buf = data[start + lo:start + hi]
        if dtype == "BOOL" and any(b not in (0, 1) for b in buf):
            raise ValueError("native physical oracle bool bytes invalid")
        value = np.frombuffer(buf, dtype="<f4" if dtype == "F32" else np.bool_).reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite native target tensor")
        result[name] = value
        spans.append((lo, hi))
    cursor = 0
    for lo, hi in sorted(spans):
        if lo != cursor:
            raise ValueError("overlap/gap in native target safetensors")
        cursor = hi
    if cursor + start != len(data):
        raise ValueError("unaccounted native target bytes")
    return result


def raw_tensor_sha(value):
    h = hashlib.sha256(A.canonical({"dtype": "torch.float32", "shape": list(value.shape)}))
    h.update(value.tobytes())
    return h.hexdigest()


def verify_arrays(value, surface, receipt, source):
    if (surface.shape != (1, 81920, 7) or surface.dtype != np.float32 or
        not np.isfinite(surface).all() or np.any(surface[:, :, 6]) or
        not np.allclose(np.linalg.norm(surface[:, :, 3:6], axis=-1), 1, atol=.002)):
        raise ValueError("actual captured original native surface differs")
    raw, model = value["clean_raw_latent"], value["clean_model_space_latent"]
    scale = receipt["scale_factor"]
    if (scale != 1.0039506158752403 or not np.array_equal(raw * np.float32(scale), model) or
        not np.allclose(model / np.float32(scale), raw, atol=2e-7, rtol=2e-7) or
        raw_tensor_sha(raw) != receipt["raw_target_sha256"]):
        raise ValueError("genuine raw/model-space VAE target mapping differs")
    # The hook-recorded target query must really be a member of the captured
    # encoder source. It is never exposed by the input index below.
    xyz = np.ascontiguousarray(surface[0, :, :3]).view("V12").ravel()
    query = np.ascontiguousarray(value["target_only_encoder_query_xyz"][0]).view("V12").ravel()
    if not np.isin(query, xyz).all():
        raise ValueError("target encoder query is not from captured actual surface")
    vertices, faces, _ = G.mesh(Path(source["geometry_target"]["mesh"]["path"]))
    _, transform = G.normalize(vertices)
    # Preserve the original normalize_mesh arithmetic order (1 / extent * 2 *
    # scale), which differs by up to two float64 ulps from the operator render's
    # mathematically equivalent 2 * scale / extent. This is source-audited.
    center = (vertices.max(0) + vertices.min(0)) / 2
    vertices = (vertices - center) * (1 / np.ptp(vertices, axis=0).max() * 2 * .9999)
    control = receipt["parent_surface_control"]
    if (A.digest(transform) != A.digest(source["geometry_target"]["surface"]["normalization"]) or
        not np.array_equal(vertices.min(0), control["normalized_bbox_min"]) or
        not np.array_equal(vertices.max(0), control["normalized_bbox_max"])):
        raise ValueError("actual native normalization frame differs")
    queries, inside = H.convex_oracle(vertices, faces)
    if (not np.array_equal(value["physical_oracle_queries"][0], queries) or
        not np.array_equal(value["physical_oracle_inside"], inside) or
        A.digest(H.oracle_metrics(value["native_geometry_logits"][0, :, 0], inside)) != A.digest(receipt["oracle_confusion"])):
        raise ValueError("independent physical mesh oracle/outcome differs")
    return receipt["oracle_confusion"]["positive_logit_as_inside"]


def verify(*, preparation, preparation_sha256, target_manifest, target_manifest_sha256,
           conditioning, conditioning_sha256, parity, parity_sha256, plan, output):
    if output.exists():
        raise ValueError("choose a fresh verified target intersection namespace")
    prep, streams = C.preparation(preparation, preparation_sha256)
    native = seal(target_manifest, target_manifest_sha256, S.SESSION)
    if (native.get("sealed") is not True or native.get("complete_source_traversal") is not True or
        native.get("device") != "cpu" or native.get("threads") != 8 or native.get("rows") != 24 or
        native.get("original_weights_unchanged") is not True or native.get("whole_checkpoint_reverified_after") is not True or
        native["initial_vae_full_state_sha256"] != native["final_vae_full_state_sha256"] or
        native["weights"] != {"bytes": H.WEIGHT_BYTES, "sha256": H.WEIGHT_SHA} or
        native.get("tool_pins") != S.tool_pins() or
        any(native["source_pins"][k]["sha256"] != v for k, v in H.PINS.items()) or
        native.get("native_train_ready") is not False or native.get("training_started") is not False):
        raise ValueError("actual complete native target parent/source/state proof required")
    target_plan = S.validate_plan(plan, native["plan"]["sha256"])
    if target_plan["preparation"]["sha256"] != preparation_sha256:
        raise ValueError("native target plan/immutable preparation differs")
    c = seal(conditioning, conditioning_sha256, C.SCHEMA)
    if (c.get("sealed") is not True or c.get("complete_source_traversal") is not True or
        c["preparation"]["sha256"] != preparation_sha256 or c.get("native_train_ready") is not False):
        raise ValueError("full original source-bound raw image preparation required")
    if A.file_sha(parity) != parity_sha256:
        raise ValueError("actual C11 Torch parity file pin differs")
    proof = json.loads(parity.read_text(encoding="utf-8"))
    if (proof.get("schema") != "recovery_geometry_conditioning_cpu_parity.v1" or
        proof.get("ingress_manifest_sha256") != conditioning_sha256 or proof.get("source_pins") != C.PINS or
        proof.get("actual_original_patchify_executed") is not True or proof.get("full_source_traversal") is not True or
        proof.get("teacher_features_generated") is not False):
        raise ValueError("actual unchanged original C11 Torch parity proof required")
    path = collected_path(conditioning.parent, c["outputs"]["receipts"])
    input_rows = {r["id"]: r for r in A.rows(path)}
    parity_rows = {r["id"]: r for r in proof["checks"]}
    native_rows = {r["id"]: r for r in native["targets"]}
    planned = {r["id"]: r for r in target_plan["sources"]}
    if (len(input_rows) != 24 or len(proof["checks"]) != 24 or len(native["targets"]) != 24 or
        any(set(index) != set(streams["source_ready"]) for index in (input_rows, parity_rows, native_rows, planned))):
        raise ValueError("full source/image/parity/target identity sets must match")
    records, families = [], defaultdict(list)
    for item_id, source in streams["source_ready"].items():
        admission = streams["admission"][item_id]
        directory = target_manifest.parent / item_id
        receipt_path = collected_path(directory, native_rows[item_id])
        r = seal(receipt_path, native_rows[item_id]["sha256"], "recovery_hy3d_clean_target.v1", "receipt_sha256")
        row_hash = A.digest(source)
        p, q = input_rows[item_id], parity_rows[item_id]
        if (r["id"] != item_id or any(x["source_row_sha256"] != row_hash for x in (r, p, q)) or
            r["source_admission_sha256"] != A.digest(admission) or
            r["source_group_id"] != source["provenance"]["group_id"] or
            r["source_mesh_sha256"] != source["geometry_target"]["mesh"]["sha256"] or
            any(x["split"] != source["split"] for x in (r, p, q, native_rows[item_id])) or
            any(x != admission["train_admitted"] for x in (r["source_train_eligible"], p["train_source_eligible"])) or
            r.get("sample_seed") != S.SEED or r.get("new_seed_selection") is not False or
            r.get("encoder") != "unchanged original posterior.sample()" or
            r.get("target_query_xyz_generation_input_allowed") is not False or
            r.get("native_train_ready") is not False or r.get("geometry_quality_admission_granted") is not False or
            (planned[item_id]["repeat_sample"] and r.get("repeated_sample_and_query_bitwise_equal") is not True) or
            q.get("raw_pixel_patch_grid_order_positions_bitwise_equal") is not True or
            r["conditioning_image_sha256"] != p["conditioning_image"]["sha256"] or
            q["asset_sha256"] != p["conditioning_image"]["sha256"]):
            raise ValueError("native clean target/source/admission/held/image provenance differs")
        for asset in (source["geometry_target"]["mesh"], source["geometry_target"]["surface"], p["conditioning_image"]):
            A.pinned(asset, "actual physical source asset")
        target_path = collected_path(directory, r["clean_target"])
        surface_path = collected_path(directory, r["parent_surface_control"]["surface"])
        dense = r["dense_reconstruction"]
        if dense.get("extracted") is not True or dense.get("resolution") != planned[item_id]["dense_resolution"]:
            raise ValueError("actual bounded native dense reconstruction missing")
        mesh_path = collected_path(directory, dense["mesh"])
        surface = np.load(surface_path, allow_pickle=False)
        oracle = verify_arrays(targets(target_path), surface, r, source)
        ingress = conditioning.parent / "shards" / Path(p["raw_image_ingress"]["path"]).name
        if A.file_sha(ingress) != p["raw_image_ingress"]["sha256"]:
            raise ValueError("complete source-bound raw C11 input file differs")
        families[r["family"]].append({"oracle": oracle, "resolution": dense["resolution"], "metrics": dense["metrics"]})
        records.append({"schema": SCHEMA, "id": item_id, "source_row_sha256": row_hash,
            "source_admission_sha256": A.digest(admission), "source_group_id": r["source_group_id"],
            "recovery_group_id": admission["recovery_group_id"], "family": r["family"], "split": source["split"],
            "source_train_eligible": admission["train_admitted"], "prepared_clean_target": True,
            "inputs": {"raw_image_ingress": {"path": str(ingress.resolve()), "sha256": A.file_sha(ingress)},
                       "conditioning_image": p["conditioning_image"]},
            "target_only": {"latent": {"path": str(target_path), "sha256": A.file_sha(target_path)},
                "native_surface": {"path": str(surface_path), "sha256": A.file_sha(surface_path)},
                "native_mc_mesh": {"path": str(mesh_path), "sha256": A.file_sha(mesh_path)},
                "native_receipt": {"path": str(receipt_path), "sha256": A.file_sha(receipt_path)},
                "encoder_query_xyz_generation_input_allowed": False},
            "oracle": oracle, "native_train_ready": False, "geometry_quality_admission_granted": False})
    output.mkdir(parents=True)
    ready = output / "prepared_targets.jsonl"
    ready.write_bytes(b"".join(A.canonical(x) + b"\n" for x in records))
    summary = {}
    for family, checks in families.items():
        summary[family] = {"rows": len(checks), "minimum_inside_recall": min(x["oracle"]["inside_recall"] for x in checks),
            "minimum_balanced_accuracy": min(x["oracle"]["balanced_accuracy"] for x in checks),
            "by_native_mc_resolution": {str(res): {"rows": sum(x["resolution"] == res for x in checks),
                "sampled_chamfer_l1_min": min(x["metrics"]["chamfer_l1_sum_of_means"] for x in checks if x["resolution"] == res),
                "sampled_chamfer_l1_max": max(x["metrics"]["chamfer_l1_sum_of_means"] for x in checks if x["resolution"] == res)}
                for res in sorted({x["resolution"] for x in checks})}}
    result = {"schema": SCHEMA, "sealed": True, "source_year": 2026, "complete_source_traversal": True,
        "sources": {name: {"path": str(p.resolve()), "sha256": h} for name, p, h in (
            ("geometry_preparation", preparation, preparation_sha256), ("native_target_session", target_manifest, target_manifest_sha256),
            ("raw_image_conditioning", conditioning, conditioning_sha256), ("original_C11_Torch_parity", parity, parity_sha256),
            ("native_target_plan", plan, native["plan"]["sha256"]))},
        "rows": len(records), "source_train_eligible": sum(r["source_train_eligible"] for r in records),
        "held_sources_retained": sum(r["split"] != "train" for r in records), "family_quality": summary,
        "outputs": {"prepared_targets": {"path": str(ready.resolve()), "sha256": A.file_sha(ready), "rows": len(records)}},
        "source_code_sha256": A.file_sha(Path(__file__)), "native_train_ready": False,
        "geometry_quality_admission_granted": False, "training_started": False,
        "blockers": ["Explicit selected query-chart/time-conversion/read-policy/supervised-flow amendment and native integration proof required.",
                     "Per-source geometric quality admission is separate; coarse native MC metrics are measured, not full natural-object coverage."]}
    result["manifest_sha256"] = A.digest(result)
    (output / "manifest.json").write_bytes(A.canonical(result) + b"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "target-manifest", "conditioning", "parity"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(**vars(args))
    print(json.dumps({k: result[k] for k in ("rows", "source_train_eligible", "held_sources_retained", "native_train_ready", "family_quality")}))


if __name__ == "__main__":
    main()
