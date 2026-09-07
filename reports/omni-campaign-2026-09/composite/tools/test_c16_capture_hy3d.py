"""Tiny SYNTHETIC observer/loading tests; never native-model capability evidence."""
import hashlib
from contextlib import contextmanager
import json
import struct
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
import c16_capture_hy3d as capture


def fps(src, batch, ratio):
    # Explicitly synthetic deterministic selector, not a replacement used by CLI.
    return torch.stack([torch.nonzero(batch == b).flatten()[[1, 3]] for b in batch.unique()]).flatten()


class SyntheticEncoder(nn.Module):
    pc_size, pc_sharpedge_size, num_latents = 8, 0, 2

    def sample_points_and_latents(self, pc, feats=None):
        b, n, _ = pc.shape
        ids = torch.randperm(n, device=pc.device)
        points = pc[:, ids].reshape(b * n, 3)
        selected = fps(points, torch.arange(b).repeat_interleave(n), ratio=0.25)
        queries = points[selected].reshape(b, 2, 3)
        return queries, points, [queries]

    def forward(self, pc, feats):
        return self.sample_points_and_latents(pc, feats)[0]


class SyntheticGeo(nn.Module):
    label_type = "binary"

    def __init__(self):
        super().__init__()
        self.chunks = []

    def forward(self, *, queries, latents):
        self.chunks.append(queries.shape[1])
        return queries.sum(-1, keepdim=True) + latents.mean()


class SyntheticVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SyntheticEncoder()
        self.geo_decoder = SyntheticGeo()
        self.sampled = None

    def encode(self, surface, sample_posterior=True):
        self.sampled = sample_posterior
        return self.encoder(surface[..., :3], surface[..., 3:])[..., :2]

    def decode(self, latents):
        return latents * 2


@pytest.fixture
def tiny_dims(monkeypatch):
    monkeypatch.setattr(capture, "CAPTURE_DIMS", {"surface_points": 8, "tokens": 2, "channels": 2})


def test_sampler_indices_survive_duplicate_coordinates_and_batches():
    encoder = SyntheticEncoder()
    points = torch.zeros(2, 8, 3)
    points[1] = 9
    native_torch, native_fps = globals()["torch"], globals()["fps"]
    torch.manual_seed(89)
    plain = encoder(points, None)
    torch.manual_seed(89)
    with capture.observe_sampler(encoder) as trace:
        observed = encoder(points, None)
    assert torch.equal(plain, observed)
    expected = trace["sampler_permutation"][torch.tensor([[1, 3], [1, 3]])]
    assert torch.equal(trace["query_source_index"], expected)
    assert torch.equal(trace["anchor_xyz"], points.gather(1, expected[..., None].expand(-1, -1, 3)))
    assert globals()["torch"] is native_torch and globals()["fps"] is native_fps
    assert "sample_points_and_latents" not in encoder.__dict__


def test_observer_restores_after_failure():
    encoder = SyntheticEncoder()
    native_torch, native_fps = globals()["torch"], globals()["fps"]
    with pytest.raises(RuntimeError, match="intentional"):
        with capture.observe_sampler(encoder):
            raise RuntimeError("intentional")
    assert globals()["torch"] is native_torch and globals()["fps"] is native_fps
    assert "sample_points_and_latents" not in encoder.__dict__


def test_sampler_refuses_unreviewed_sharpedge():
    encoder = SyntheticEncoder()
    encoder.pc_sharpedge_size = 1
    with pytest.raises(ValueError, match="sharpedge"):
        with capture.observe_sampler(encoder):
            pass


def test_sampler_refuses_cross_batch_indices(monkeypatch):
    monkeypatch.setitem(globals(), "fps", lambda *args, **kwargs: torch.tensor([1, 3, 2, 4]))
    with pytest.raises(ValueError, match="batch/token"):
        encoder = SyntheticEncoder()
        with capture.observe_sampler(encoder):
            encoder(torch.randn(2, 8, 3), None)


def test_posterior_mode_and_chunked_native_binary_readout(tiny_dims):
    vae = SyntheticVAE().eval().requires_grad_(False)
    surface = torch.randn(1, 8, 7)
    payload, posterior = capture.capture_posterior(vae, surface, 3, query_chunk=1)
    assert vae.sampled is False
    assert vae.geo_decoder.chunks == [1, 1]
    assert torch.equal(posterior, payload["posterior_mode"])
    assert torch.equal(payload["anchor_xyz"], payload["field_queries"])
    assert payload["parent_field_logits"].shape == (1, 2, 1)
    assert not any("sdf" in key for key in payload)
    assert all(v.device.type == "cpu" and not v.requires_grad for v in payload.values())


@pytest.mark.parametrize("bad", ["train", "field", "surface", "chunk"])
def test_posterior_refuses_unreviewed_inputs(tiny_dims, bad):
    vae = SyntheticVAE().eval()
    surface, chunk = torch.randn(1, 8, 7), 1
    if bad == "train":
        vae.train()
    elif bad == "field":
        vae.geo_decoder.label_type = "sdf"
    elif bad == "surface":
        surface[0, 0, 0] = float("nan")
    else:
        chunk = 0
    with pytest.raises(ValueError):
        capture.capture_posterior(vae, surface, 3, chunk)


class SyntheticTime(nn.Module):
    def forward(self, sigma):
        return sigma[:, None, None].expand(-1, 1, 4)


class SyntheticAttention(nn.Module):
    def forward(self, x, context=None):
        return x * 0.01 + (0 if context is None else context.mean() * 0.02)


class SyntheticGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4))

    def forward(self, x):
        scores = F.linear(x.flatten(0, 1), self.weight).softmax(-1)
        weights, ids = scores.topk(2, sorted=False)
        return ids, weights, None


class SyntheticMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = SyntheticGate()
        self.shared_experts = nn.Linear(4, 4)

    def forward(self, x):
        _, weights, _ = self.gate(x)
        return x * weights.sum(-1).reshape(*x.shape[:2], 1) + self.shared_experts(x)


class SyntheticBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index, self.use_moe = index, index >= 15
        self.attn1, self.attn2 = SyntheticAttention(), SyntheticAttention()
        if self.use_moe:
            self.moe = SyntheticMoE()
        else:
            self.mlp = nn.Linear(4, 4)

    def forward(self, x, c, text_states, skip_value=None):
        if skip_value is not None:
            x = x + skip_value * 0.01
        x = x + self.attn1(x)
        x = x + self.attn2(x, text_states)
        return x + (self.moe(x) if self.use_moe else self.mlp(x)) * 0.01


class SyntheticDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embedder = SyntheticTime()
        self.blocks = nn.ModuleList([SyntheticBlock(i) for i in range(21)])

    def forward(self, x, sigma, contexts):
        time = self.t_embedder(sigma)
        x = torch.cat([time, x], 1)
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x, time, contexts["main"], skip_value=skips.pop() if i > 10 else None)
            if i < 10:
                skips.append(x)
        return x[:, 1:]


@pytest.mark.parametrize("layer", [0, 10, 11, 14, 15, 20])
def test_full_dit_observer_keeps_self_cross_time_skip_and_moe(layer):
    torch.manual_seed(45)
    model = SyntheticDiT().eval().requires_grad_(False)
    x, sigma, contexts = torch.randn(2, 2, 4), torch.tensor([0.2, 0.8]), {"main": torch.randn(2, 3, 4)}
    with torch.inference_mode():
        expected = model(x, sigma, contexts)
    original_f = globals()["F"]
    evidence, observed = capture.capture_dit(model, x, sigma, contexts, layer)
    assert torch.equal(expected, observed)
    assert evidence["block_input"].shape == (2, 3, 4)
    assert evidence["self_attention_write"].shape == (2, 3, 4)
    assert evidence["cross_attention_write"].shape == (2, 3, 4)
    assert torch.equal(evidence["time_token"], sigma[:, None, None].expand(-1, 1, 4))
    assert ("skip_value" in evidence) == (layer > 10)
    assert ("shared_output" in evidence) == (layer >= 15)
    if layer >= 15:
        assert torch.equal(evidence["router_topk_weights"], evidence["router_logits"].softmax(-1).gather(-1, evidence["router_topk_ids"]))
        assert bool((evidence["router_topk_weights"].sum(-1) < 1).all())
    assert globals()["F"] is original_f
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_dit_hook_and_function_namespace_restoration_on_exception():
    model = SyntheticDiT().eval().requires_grad_(False)
    original_f = globals()["F"]
    model.blocks[20].forward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("intentional"))
    with pytest.raises(RuntimeError, match="intentional"):
        capture.capture_dit(model, torch.randn(1, 2, 4), torch.tensor([0.5]), {"main": torch.randn(1, 2, 4)}, 15)
    assert globals()["F"] is original_f
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_strict_safe_combined_load_only_selected_component(tmp_path):
    parent = tmp_path / "combined.safetensors"
    weight, bias = torch.randn(3, 2).half(), torch.randn(3).half()
    save_file({"vae.weight": weight, "vae.bias": bias, "model.unrelated": torch.zeros(2).half()}, str(parent))
    result = capture.strict_component(nn.Linear(2, 3), parent, "vae.", "cpu")
    assert torch.equal(result.weight, weight) and torch.equal(result.bias, bias)
    assert result.weight.dtype == torch.float16
    assert not result.training and not any(p.requires_grad for p in result.parameters())


@pytest.mark.parametrize("bad", ["missing", "extra", "shape", "dtype", "nonfinite", "pickle"])
def test_strict_loader_refusals(tmp_path, bad):
    parent = tmp_path / "combined.safetensors"
    data = {"vae.weight": torch.ones(3, 2).half(), "vae.bias": torch.ones(3).half()}
    if bad == "missing":
        del data["vae.bias"]
    elif bad == "extra":
        data["vae.extra"] = torch.ones(1).half()
    elif bad == "shape":
        data["vae.weight"] = torch.ones(3, 4).half()
    elif bad == "dtype":
        data["vae.weight"] = data["vae.weight"].float()
    elif bad == "nonfinite":
        data["vae.bias"][0] = float("inf")
    save_file(data, str(parent))
    if bad == "pickle":
        parent.write_bytes(b"PK\x03\x04" + b"x" * 100)
    with pytest.raises(ValueError):
        capture.strict_component(nn.Linear(2, 3), parent, "vae.", "cpu")


def write_test_mesh(path, vertices, faces, comment="SYNTHETIC fixture"):
    """Tiny real-format fixtures, never parent/child modality evidence."""
    if path.suffix == ".ply":
        header = (f"ply\nformat ascii 1.0\ncomment {comment}\nelement vertex {len(vertices)}\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        rows = [" ".join(map(str, row)) for row in vertices]
        rows += ["3 " + " ".join(map(str, row)) for row in faces]
        path.write_text(header + "\n".join(rows) + "\n", encoding="ascii")
    else:
        data = comment.encode("ascii")[:80].ljust(80, b" ") + struct.pack("<I", len(faces))
        for face in faces:
            xyz = np.asarray(vertices)[face].reshape(-1).tolist()
            data += struct.pack("<12fH", 0, 0, 0, *xyz, 0)
        path.write_bytes(data)


def object_manifest(tmp_path):
    pytest.importorskip("trimesh", reason="genuine parser fixture needs native runtime dependency")
    rows = []
    for i, split in enumerate(("fit", "held")):
        mesh, image = tmp_path / f"object-{i}.ply", tmp_path / f"image-{i}.png"
        write_test_mesh(mesh, [[0, 0, 0], [i + 1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        image.write_bytes(f"SYNTHETIC IMAGE ID {i}".encode())
        rows.append({"object_file": mesh.name, "object_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                     "conditioning_file": image.name, "conditioning_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                     "split": split, "seed": i, "sigma": 0.4})
    return {"schema": "neural-forge.hy3d-capture-objects.v1", "objects": rows}


@pytest.mark.parametrize("bad", [None, "leak", "digest", "seed", "sigma", "one_split", "path"])
def test_grouped_object_inputs(tmp_path, bad):
    body = object_manifest(tmp_path)
    rows = body["objects"]
    if bad == "leak":
        rows[1].update(object_file=rows[0]["object_file"], object_sha256=rows[0]["object_sha256"])
    elif bad == "digest":
        rows[0]["object_sha256"] = "0" * 64
    elif bad == "seed":
        rows[0]["seed"] = True
    elif bad == "sigma":
        rows[0]["sigma"] = 1000
    elif bad == "one_split":
        rows[1]["split"] = "fit"
    elif bad == "path":
        rows[0]["object_file"] = 5
    path = tmp_path / "objects.json"
    path.write_text(json.dumps(body))
    if bad is not None:
        with pytest.raises(ValueError):
            capture.read_objects(path)
    else:
        result = capture.read_objects(path)
        assert len(result) == 2 and Path(result[0]["object_file"]).is_absolute()
        assert result[0]["geometry_identity"]["schema"] == capture.GEOMETRY_SCHEMA


def test_canonical_geometry_ignores_serialization_and_index_order_without_mutation():
    vertices = np.array([[0., 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    original = vertices.copy()
    expected = capture.canonical_geometry(vertices, faces)
    permutation = np.array([3, 1, 0, 2])
    changed = vertices[permutation].copy()
    changed[changed == 0] = -0.0
    remapped = np.argsort(permutation)[faces][::-1, ::-1]
    # Duplicated vertex, unused vertex, duplicate/reversed face all describe
    # the same triangle set even though the serialized artifact differs.
    changed = np.concatenate([changed, changed[2:3], [[99, 99, 99]]])
    remapped[remapped == 2] = 4
    remapped = np.concatenate([remapped, remapped[:1, ::-1]])
    assert capture.canonical_geometry(changed, remapped) == expected
    assert np.array_equal(vertices, original)
    altered = vertices.copy()
    altered[1, 0] = np.nextafter(2., 3.)
    assert capture.canonical_geometry(altered, faces) != expected  # no tolerance/rounding
    assert capture.canonical_geometry(vertices, [[0, 1, 3], [0, 2, 3]]) != expected


@pytest.mark.parametrize("bad", ["nan", "inf", "empty", "nontriangle", "negative", "outside", "float_indices", "bound"])
def test_canonical_geometry_refusals(monkeypatch, bad):
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    if bad in ("nan", "inf"):
        vertices[0, 0] = float(bad)
    elif bad == "empty":
        faces = np.empty((0, 3), dtype=int)
    elif bad == "nontriangle":
        faces = np.array([[0, 1]])
    elif bad == "negative":
        faces[0, 0] = -1
    elif bad == "outside":
        faces[0, 0] = 3
    elif bad == "float_indices":
        faces = faces.astype(float)
    else:
        monkeypatch.setattr(capture, "MAX_GEOMETRY_FACES", 0)
    with pytest.raises(ValueError):
        capture.canonical_geometry(vertices, faces)


@pytest.mark.parametrize("extension", [".ply", ".stl"])
def test_genuine_parser_rejects_duplicate_geometry_across_splits(tmp_path, extension):
    body = object_manifest(tmp_path)
    rows = body["objects"]
    duplicate = tmp_path / ("different-name" + extension)
    write_test_mesh(duplicate, [[0, 1, 0], [0, 0, 0], [1, 0, 0]], [[2, 1, 0]], "changed metadata")
    rows[1].update(object_file=duplicate.name, object_sha256=capture.digest_file(duplicate))
    assert rows[0]["object_sha256"] != rows[1]["object_sha256"]
    path = tmp_path / "objects.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="identical geometry leaked"):
        capture.read_objects(path)
    # The same geometry may appear in several views/timesteps in ONE split.
    rows[1]["split"] = "fit"
    distinct = tmp_path / "distinct.ply"
    write_test_mesh(distinct, [[0, 0, 0], [3, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    rows.append({**rows[0], "object_file": distinct.name, "object_sha256": capture.digest_file(distinct), "split": "held"})
    path.write_text(json.dumps(body))
    result = capture.read_objects(path)
    assert result[0]["geometry_identity"] == result[1]["geometry_identity"]
    assert result[0]["geometry_identity"] != result[2]["geometry_identity"]


def test_declared_geometry_cannot_override_computed_identity(tmp_path):
    body = object_manifest(tmp_path)
    body["objects"][0]["geometry_identity"] = {"sha256": "0" * 64}
    path = tmp_path / "objects.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="declared geometry identity mismatch"):
        capture.read_objects(path)


def test_geometry_parser_is_bound_to_actual_bytes(tmp_path):
    body = object_manifest(tmp_path)
    row = body["objects"][0]
    mesh = tmp_path / row["object_file"]
    with pytest.raises(ValueError, match="geometry input digest mismatch"):
        capture.geometry_identity(mesh, "0" * 64)
    mesh.write_text("not a PLY file")
    with pytest.raises(ValueError, match="geometry input digest mismatch"):
        capture.geometry_identity(mesh, row["object_sha256"])


def test_sealed_artifacts_refuse_overwrite_and_nonfinite(tmp_path):
    path = tmp_path / "sample.safetensors"
    receipt = capture.sealed_tensors(path, {"x": torch.tensor([1.0])})
    assert receipt["file_sha256"] == capture.digest_file(path)
    assert torch.equal(load_file(str(path))["x"], torch.tensor([1.0]))
    with pytest.raises(ValueError, match="replace"):
        capture.sealed_tensors(path, {"x": torch.tensor([2.0])})
    with pytest.raises(ValueError, match="nonfinite"):
        capture.sealed_tensors(tmp_path / "bad.safetensors", {"x": torch.tensor([float("nan")])})
    capture.sealed_json(tmp_path / "receipt.json", receipt)
    with pytest.raises(FileExistsError):
        capture.sealed_json(tmp_path / "receipt.json", receipt)


def test_native_namespace_refuses_source_changes_without_importing(tmp_path):
    root = tmp_path / capture.PACKAGE_DIR
    root.mkdir(parents=True)
    (root / "preprocessors.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="re-audit"):
        with capture.native_modules(tmp_path):
            pass
    assert not any(n.startswith("_nf_hy3d_capture_native") for n in sys.modules)


@pytest.mark.parametrize("fail", [False, True, "replace_input"])
def test_synthetic_cli_posterior_path_and_partial_preservation(tmp_path, tiny_dims, monkeypatch, fail):
    # Entire native dependency boundary is explicitly synthetic in this unit test.
    # The CLI itself has no injection/fixture switch.
    manifest = tmp_path / "objects.json"
    manifest.write_text(json.dumps(object_manifest(tmp_path)))
    parent = tmp_path / "parent.safetensors"
    save_file({"synthetic": torch.zeros(1)}, str(parent))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "dit_config_2_1.yaml").write_text('{"vae":{"params":{}}}')
    vae = SyntheticVAE().eval().requires_grad_(False)
    vae.scale_factor = 1.0039506158752403
    surface = torch.arange(56).reshape(1, 8, 7).float() / 56
    loaded = []

    def load_surface(path):
        loaded.append(path)
        if fail is True and len(loaded) == 2:
            raise RuntimeError("intentional second object failure")
        return surface.clone()

    @contextmanager
    def modules(root, include_dit):
        assert include_dit is False
        yield {"vae": SimpleNamespace(ShapeVAE=lambda **kwargs: vae),
               "surface": SimpleNamespace(SharpEdgeSurfaceLoader=lambda **kwargs: load_surface),
               "extra_source_sha256": {}}

    monkeypatch.setattr(capture, "native_modules", modules)
    def strict_component(module, *args):
        if fail == "replace_input":
            (tmp_path / "object-0.ply").write_text("replaced during model initialization")
        return module

    monkeypatch.setattr(capture, "strict_component", strict_component)
    monkeypatch.setattr(capture, "inspect_source", lambda *args: {"source_identity": {"synthetic": True}})
    monkeypatch.setattr(capture, "verify_native_dependencies", lambda *args: None)
    output = tmp_path / "out"
    argv = ["--source-root", str(tmp_path), "--parent-checkpoint", str(parent), "--objects", str(manifest),
            "--output", str(output), "--device", "cpu", "--query-chunk", "1"]
    if fail == "replace_input":
        with pytest.raises(ValueError, match="capture input digest changed"):
            capture.main(argv)
        assert loaded == []
        assert not (output / "capture-run.json").exists()
        return
    if fail:
        with pytest.raises(RuntimeError, match="second object"):
            capture.main(argv)
        assert (output / "sample-0000.safetensors").is_file()
        assert not (output / "capture-run.json").exists()
    else:
        assert capture.main(argv) == 0
        report = json.loads((output / "capture-run.json").read_text())
        assert report["d10_sdf_pairs_complete"] is False
        assert report["folded_child_proven"] is False
        assert report["dit_capture"] is None
        assert len(report["samples"]) == 2
        assert report["field_kind"] == "native_binary_field_logits_not_metric_sdf"
        for receipt in report["samples"]:
            assert capture.digest_file(output / receipt["file"]) == receipt["file_sha256"]
            assert receipt["geometry_identity"]["schema"] == capture.GEOMETRY_SCHEMA
    with pytest.raises(ValueError, match="fresh output"):
        capture.main(argv)
