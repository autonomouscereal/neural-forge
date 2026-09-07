"""Explicitly synthetic CPU mechanics; never authentic media/capability evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import c16_h3_patch_contract as m


def shift(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1 - from_shift))
    return to_shift * base / (1 + (to_shift - 1) * base)


def spans(n):
    return [(1, 4, 4, 4, 4)[i % 5] * 5 / 3 for i in range(n)]


def patchify(x):
    _, c, t, h, w = x.shape
    return x.reshape(1, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, c * 4)


def audio_pack(x):
    return x[0].permute(1, 2, 0).reshape(-1, 32)


def layout(text, t, h, w, audio):
    # Independent tiny layout oracle for unit tests, not a native source claim.
    hh = (torch.arange(h // 2, dtype=torch.float64) * (h / (h * w) ** .5 / (h // 2)) + (1 - h / (h * w) ** .5) / 2) * 32
    ww = (torch.arange(w // 2, dtype=torch.float64) * (w / (h * w) ** .5 / (w // 2)) + (1 - w / (h * w) ** .5) / 2) * 32
    coords = [[i, 0, 0] for i in range(text)]
    coords += [[text + i, 0, float(ww[channel * (len(ww) - 1)])] for channel in range(2) for i in range(audio)]
    ts = torch.cat([torch.zeros(1, dtype=torch.float64), torch.tensor(spans(t)[:-1], dtype=torch.float64).cumsum(0)])
    coords += [[text + float(ts[f]), float(y), float(x)] for f in range(t) for y in hh for x in ww]
    return SimpleNamespace(seq_len=len(coords), position_ids=torch.tensor(coords, dtype=torch.float64))


@pytest.fixture
def example():
    generator = torch.Generator().manual_seed(21)
    normal = lambda shape: torch.randn(shape, generator=generator)
    vshape, ashape = [1, 24, 7, 2, 4], [1, 32, 2, round(sum(spans(7)))]
    meta = {"route": "plain_t2va", "minimax_payload": {}, "transformer_options": {},
            "stimuli": "s1_media", "deviations": [], "compute_dtype": "float32", "allow_tf32": False,
            "video_shape": vshape, "audio_shape": ashape, "text_len": 2, "raw_frames": 22}
    state = {"timestep": torch.tensor([500.]), "sigma_video": torch.tensor([.5]),
             "video_latent_frame_ids": torch.arange(7)}
    state["sigma_audio"] = shift(state["sigma_video"], 12., 3.)
    for band, shape in (("video", vshape), ("audio", ashape)):
        state[band + "_clean"], state[band + "_noise"] = normal(shape), normal(shape)
        sigma = state["sigma_" + band]
        state[band + "_x"] = (1 - sigma) * state[band + "_clean"] + sigma * state[band + "_noise"]
    state["video_patch_input"], state["audio_patch_input"] = patchify(state["video_x"]), audio_pack(state["audio_x"])
    weights = {key: normal(shape) / 10 for key, shape in m.PROJECTIONS.items()}
    for band in ("video", "audio"):
        state[band + "_patch_output"] = F.linear(state[band + "_patch_input"], weights[band + "_patch_proj.weight"], weights[band + "_patch_proj.bias"])
    state["text_states"], state["text_refined"] = normal([1, 2, 5120]), normal([2, 5376])
    state["block0_input"] = torch.cat([state["text_refined"], state["audio_patch_output"], state["video_patch_output"]])
    state["position_ids"] = layout(2, 7, 2, 4, ashape[-1]).position_ids
    owner = [0] + [1] * 3 + [2] * 4 + [3] * 5 + [4] * 4 + [5] + [6] * 4
    influence = F.one_hot(torch.tensor(owner), 7).double()
    probe = {"frames": 22, "latent_frames": 7, "owner_per_frame": owner, "influence": influence.tolist(),
             "groups": {str(i): [j for j, value in enumerate(owner) if value == i] for i in range(7)}}
    geometry = {"patchify_video": patchify, "pack_audio": audio_pack, "PackedLayout": layout,
                "_video_t_spans": spans, "time_shift_sigma": shift}
    return meta, state, geometry, weights, probe


def test_full_native_widths_and_nonuniform_all_frame_coverage(example):
    report = m.validate_forward(*example)
    assert report["video_patch_width"] == 96 and report["audio_patch_width"] == 32
    assert report["temporal_coverage"]["all_latent_frames_required"] == list(range(7))
    assert report["temporal_coverage"]["group_sizes"] == [1, 3, 4, 5]
    assert report["parent_dit_forward_replayed"] is False
    assert report["projection_rel_l2"] == {"video": 0., "audio": 0.}


@pytest.mark.parametrize("change,match", [
    (lambda a: a[0].update(deviations=["gaussian text"]), "deviated"),
    (lambda a: a[0].update(stimuli="procedural"), "deviated"),
    (lambda a: a[0].update(minimax_payload={"refs": [{}]}), "plain T2VA"),
    (lambda a: a[0].update(transformer_options={"minimax_h3_sigma_shift_audio": 12}), "overrides"),
    (lambda a: a[0].update(allow_tf32=True), "FP32 compute"),
    (lambda a: a[0]["video_shape"].__setitem__(3, 3), "silent padding"),
    (lambda a: a[0]["audio_shape"].__setitem__(2, 1), "stereo"),
    (lambda a: a[0]["audio_shape"].__setitem__(3, 80), "audio span"),
    (lambda a: a[1].update(video_latent_frame_ids=torch.tensor([1, 2, 3, 4, 6])), "video_latent_frame_ids"),
    (lambda a: a[1]["video_latent_frame_ids"].add_(1), "every original latent"),
    (lambda a: a[1].update(video_patch_input=a[1]["video_patch_input"][:, :24]), "video_patch_input"),
    (lambda a: a[1].update(audio_patch_input=a[1]["audio_patch_input"].flip(0)), "audio native patch"),
    (lambda a: a[1].update(video_patch_input=a[1]["video_patch_input"].flip(1)), "video native patch"),
    (lambda a: a[1].update(position_ids=a[1]["position_ids"].float()), "position_ids"),
    (lambda a: a[1]["position_ids"][:, 0].add_(1), "THW positions"),
    (lambda a: a[1]["video_patch_output"].add_(1), "projection replay"),
    (lambda a: a[1]["block0_input"].add_(1), "packed block0"),
    (lambda a: a[1]["sigma_audio"].add_(.01), "shifted sigma"),
    (lambda a: a[1]["timestep"].add_(1), "timestep"),
    (lambda a: a[1]["video_clean"].add_(1), "noising"),
    (lambda a: a[4]["groups"].pop("0"), "omit/relabel"),
    (lambda a: a[4]["groups"].pop("5"), "omit/relabel"),
    (lambda a: a[4]["owner_per_frame"].__setitem__(0, 1), "measured argmax"),
    (lambda a: a[4].update(influence=[[0.] * 7] * 22), "zero ownership"),
])
def test_incomplete_or_changed_boundary_refuses(example, change, match):
    change(example)
    with pytest.raises(ValueError, match=match):
        m.validate_forward(*example)


def test_empty_argmax_owner_does_not_authorize_skipping_latent(example):
    probe = example[-1]
    probe["owner_per_frame"][0] = 1
    probe["influence"][0] = [0., 1., 0., 0., 0., 0., 0.]
    probe["groups"]["0"] = []
    probe["groups"]["1"] = [0, 1, 2, 3]
    assert m.validate_forward(*example)["temporal_coverage"]["all_latent_frames_required"] == list(range(7))


def ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_source_identity_is_not_a_fixture_switch(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("raise RuntimeError('must not execute')", encoding="utf-8")
    with pytest.raises(ValueError, match="source identity"):
        m.native_geometry(source)


def test_file_pin_mutation_and_hash_refusal(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("{}", encoding="utf-8")
    pins = m.Pins()
    pinned = ref(path)
    pins.file(pinned)
    path.write_text("{\"changed\":true}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        pins.stable()
    with pytest.raises(ValueError, match="hash mismatch"):
        m.Pins().file(pinned)


@pytest.mark.parametrize("fixture", [True, None, "false"])
def test_fixture_never_accepted_by_production_cli_contract(tmp_path, fixture):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": m.SCHEMA, "fixture": fixture}), encoding="utf-8")
    with pytest.raises(ValueError, match="no fixture CLI"):
        m.validate_capture(ref(path))


def test_full_parent_hash_not_attempted_for_wrong_header(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    header = b"{}"
    path.write_bytes(len(header).to_bytes(8, "little") + header)
    monkeypatch.setattr(m, "PARENT_BYTES", path.stat().st_size)
    with pytest.raises(ValueError, match="header identity"):
        m.inspect_parent(path, m.PROJECTION_SHA)


def test_only_geometry_nodes_execute_after_pin(tmp_path, monkeypatch):
    # Synthetic AST-loader unit test; production SHA and CLI have no fixture flag.
    data = "raise RuntimeError('top level must not run')\n" + "\n".join(
        f"def {name}():\n    return 17\n" for name in sorted(m.GEOMETRY))
    path = tmp_path / "synthetic_source.py"
    path.write_text(data, encoding="utf-8")
    monkeypatch.setattr(m, "SOURCE_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    result = m.native_geometry(path)
    assert all(result[name]() == 17 for name in m.GEOMETRY)


@pytest.fixture
def manifest_case(tmp_path, monkeypatch, example):
    # End-to-end descriptor mechanics with explicit test monkeypatches ONLY.
    # The shipped entry point never substitutes native source or parent weights.
    meta, state, geometry, projections, probe = example
    source = tmp_path / "synthetic-source.txt"
    source.write_text("unit fixture only", encoding="utf-8")
    monkeypatch.setattr(m, "native_geometry", lambda path: geometry)
    monkeypatch.setattr(m, "inspect_parent", lambda path, hashes: (projections, m.stamp(path)))
    manifest = {"schema": m.SCHEMA, "fixture": False, "native_source": ref(source), "producer": ref(source),
                "parent_checkpoint": str(source), "projection_raw_bytes_sha256": m.PROJECTION_SHA,
                "max_capture_bytes": 16 << 20, "forwards": []}
    for index, split in enumerate(("fit", "held")):
        state = {key: value.clone() for key, value in state.items()}
        if index:
            for band in ("video", "audio"):
                state[band + "_clean"].add_(.03)
                sigma = state["sigma_" + band]
                state[band + "_x"] = (1 - sigma) * state[band + "_clean"] + sigma * state[band + "_noise"]
                state[band + "_patch_input"] = (patchify if band == "video" else audio_pack)(state[band + "_x"])
                state[band + "_patch_output"] = F.linear(state[band + "_patch_input"], projections[band + "_patch_proj.weight"], projections[band + "_patch_proj.bias"])
            state["block0_input"] = torch.cat([state["text_refined"], state["audio_patch_output"], state["video_patch_output"]])
        row = copy.deepcopy(meta)
        row.update(id=f"forward{index}", split=split)
        path = tmp_path / f"rows{index}.safetensors"
        save_file({key: t.contiguous() for key, t in state.items()}, str(path))
        row["tensors"] = ref(path)
        for band in ("video", "audio", "text"):
            asset = tmp_path / f"{band}{index}.txt"
            asset.write_text(f"synthetic {band} asset {index}", encoding="utf-8")
            row[band + "_asset"] = ref(asset)
            row[band + "_parent_group"] = f"original-{band}-{index}"
            receipt = {"kind": "h3_native_" + band + "_encode", "actual_parent_forward": True,
                       "source_code": ref(source), "source_identity_receipt": ref(source)}
            identity = {"schema": "neural-forge.h3-encoder-identity.v1", "source_code_sha256": ref(source)["sha256"],
                        "actual_tensor_loader_verified": True, "checkpoint_path": str(source), "checkpoint_bytes": source.stat().st_size,
                        "checkpoint_header_sha256": "1" * 64,
                        "family": {"video": "minimax_h3_video_vae", "audio": "minimax_h3_audio_vae", "text": "qwen3_vl"}[band]}
            identity_path = tmp_path / f"{band}-identity{index}.json"
            identity_path.write_text(json.dumps(identity), encoding="utf-8")
            receipt["source_identity_receipt"] = ref(identity_path)
            if band == "text":
                receipt.update(encoder_family="qwen3_vl", hidden_layer=50, prompt=ref(asset),
                               output_raw_bytes_sha256=m.raw_sha(state["text_states"]))
            else:
                receipt.update(input_sha256=ref(asset)["sha256"], normalization="native_h3_model_input",
                               output_raw_bytes_sha256=m.raw_sha(state[band + "_clean"]))
            key = "text_states" if band == "text" else band + "_clean"
            receipt.update(output_shape=list(state[key].shape), output_dtype="float32")
            if band == "video":
                receipt["raw_frames"] = meta["raw_frames"]
            path = tmp_path / f"{band}-receipt{index}.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            row[band + "_encoder_receipt"] = ref(path)
        temporal = copy.deepcopy(probe)
        temporal.update(input_sha256=row["video_asset"]["sha256"], encoder_receipt_sha256=row["video_encoder_receipt"]["sha256"])
        path = tmp_path / f"probe{index}.json"
        path.write_text(json.dumps(temporal), encoding="utf-8")
        row["temporal_probe"] = ref(path)
        manifest["forwards"].append(row)
    return manifest, tmp_path


def run_case(case):
    manifest, root = case
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return m.validate_capture(ref(path))


def test_descriptor_mechanics_never_claim_encoders_or_child(manifest_case):
    result = run_case(manifest_case)
    assert [row["id"] for row in result["forwards"]] == ["forward0", "forward1"]
    assert result["contract_valid"]
    assert result["production_install_eligible"] is False
    assert result["encoders_independently_replayed"] is False


@pytest.mark.parametrize("change,match", [
    (lambda x: x["forwards"][1].update(video_parent_group=x["forwards"][0]["video_parent_group"]), "crosses fit/held"),
    (lambda x: x["forwards"][1].update(audio_parent_group=x["forwards"][0]["audio_parent_group"]), "crosses fit/held"),
    (lambda x: x["forwards"][1].update(video_asset=x["forwards"][0]["video_asset"]), "crosses fit/held"),
    (lambda x: x["forwards"][1].update(audio_asset=x["forwards"][0]["audio_asset"]), "crosses fit/held"),
    (lambda x: x["forwards"][1].update(id=x["forwards"][0]["id"]), "id missing/duplicated"),
    (lambda x: x["forwards"][0].pop("text_encoder_receipt"), "missing file reference"),
    (lambda x: x["forwards"][0].update(video_encoder_receipt=x["forwards"][0]["audio_encoder_receipt"]), "video encoder receipt"),
    (lambda x: x.update(max_capture_bytes=8), "read budget"),
    (lambda x: x["forwards"][1].update(split="fit"), "both clip-disjoint splits"),
])
def test_descriptor_rejects_missing_identity_and_joint_split_leaks(manifest_case, change, match):
    change(manifest_case[0])
    with pytest.raises(ValueError, match=match):
        run_case(manifest_case)


def test_metadata_rewrapped_identical_clean_clip_refuses(manifest_case):
    manifest, _ = manifest_case
    fit, held = manifest["forwards"]
    held["tensors"] = fit["tensors"]
    path = Path(held["video_encoder_receipt"]["path"])
    receipt = json.loads(path.read_text())
    fit_receipt = json.loads(Path(fit["video_encoder_receipt"]["path"]).read_text())
    receipt["output_raw_bytes_sha256"] = fit_receipt["output_raw_bytes_sha256"]
    path.write_text(json.dumps(receipt))
    held["video_encoder_receipt"] = ref(path)
    with pytest.raises(ValueError, match="identical encoded AV content"):
        run_case(manifest_case)
