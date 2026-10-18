"""Tests for media_b_teacher_cache.py 1.0.9 (lane media-b, Claude, 2026-10-13). Server, CPU only:

    cd /home/cereal/omni-campaign/distill/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 PYTHONPATH=$CHIMERA_OMNI_SRC/src \
    OMP_NUM_THREADS=8 nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \
        -p no:cacheprovider -q test_media_b_teacher_cache.py

Synthetic: the s7_pairs_parent 1.1.0 pin, content-addressed corpora, the shard store (write, resume, seal, verify,
and the refusals: tampered shard, missing shard, non-finite tensor, teacher changed), the HY3D flow-time convention
and Euler chain against an independent restatement of the upstream scheduler, the ACE deployed schedule against an
independent restatement of ComfyUI's ModelSamplingDiscreteFlow + 'simple' scheduler, the audio-code band top-k, the
Qwen-VL smart_resize, the P8 header reader, and the GPU guard (forbidden cards, the 2,000 MiB start rule, one card
per process). 1.0.5: the run config is part of the cache address (a --cfg / --block change is a different cache),
resume and seal refuse shards of another config or a tool version outside COMPAT, cache_config.json conflicts refuse,
pre-1.0.5 manifests do not verify, the ACE x0 stage files (round trip, tamper, keyed on x0 arguments only), the
CPU-only launch path, the s7 reference-row lookup and the manifest check roll-up. 1.0.8: the config claim on a
filesystem without hard links (exFAT: os.link EPERM), the read-write mount of a Lexar cache root (and the refusal of
any other root outside $OC), and `corpus-import` of a distill_corpus_build media set (items sha, per-file sha, schema
and kind checks). 1.0.9: `check` of a QVL cache requires a parent reference (its own s7 arms, or a sealed side cache
of the same run config, found or named), re-gates every arm's rel_fro, and fails on no reference, a failing arm, an
arm without rel_fro, another run config, a tampered side cache or a block other than 13. Real artefacts (skipped when
absent): a sealed mini-cache verifies and carries its run config; the real HY3D distill set imports to 256 items; the
s7 capture images are 512x512 and keep that side through smart_resize, so the unchanged qvl-run fires its arm on them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

HERE = Path(__file__).resolve().parent
os.environ.setdefault("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MB = _load(HERE / "media_b_teacher_cache.py", "media_b_teacher_cache_t")


# ---------------------------------------------------------------- identity


def test_version_and_builds_on_s7_1_1_0() -> None:
    assert MB.TOOL_VERSION == "1.0.9"
    assert all(MB.TOOL_VERSION in v for v in MB.COMPAT.values())
    ident = MB.tool_identity()
    assert ident["sha256"] == MB.sha256_file(HERE / "media_b_teacher_cache.py")  # import-time sha == the file on disk
    assert ident["builds_on"]["name"] == "s7_pairs_parent" and ident["builds_on"]["version"] == "1.1.0"
    assert len(ident["sha256"]) == 64


def test_other_s7_version_is_refused(tmp_path) -> None:
    src = Path(MB.load_s7().MEDIA_B_PATH).read_text(encoding="utf-8")
    fake = tmp_path / "s7_pairs_parent.py"
    fake.write_text(src.replace('TOOL_VERSION = "1.1.0"', 'TOOL_VERSION = "1.0.0"'), encoding="utf-8")
    with pytest.raises(MB.Refusal):
        MB.load_s7(fake)


def test_ace_dit_integrity_covers_the_dual_text_encoder() -> None:
    files = MB.teacher_files("ace_dit_velocity")
    assert MB.ACE_TE in files and MB.ACE_LM in files and MB.ACE_DIT in files and MB.ACE_VAE in files


def test_ace_vae_defaults_to_cpu() -> None:
    parser = MB.build_parser()
    for cmd in (["ace-dit-run", "--corpus", "c"], ["ace-probe", "--corpus", "c", "--out", "o"]):
        assert parser.parse_args(cmd).vae_device == "cpu"


def test_parser_has_every_subcommand() -> None:
    parser = MB.build_parser()
    choices = next(a for a in parser._actions if a.dest == "cmd").choices
    assert set(choices) == set(MB.WORKERS) | set(MB.HOST)


# ---------------------------------------------------------------- corpora


def _items(n: int) -> list[dict]:
    return [{"id": f"i{k}", "sha256": f"{k:064x}", "path": f"/x/{k}"} for k in range(n)]


def _hy3d_args(**kw) -> argparse.Namespace:
    extra = []
    for k, v in kw.items():
        extra += [f"--{k.replace('_', '-')}", str(v)]
    return MB.build_parser().parse_args(["hy3d-run", "--corpus", "c", *extra])


def test_corpus_sha_depends_on_items_only() -> None:
    items = _items(3)
    a = MB.corpus_sha("hy3d_images", items)
    assert a == MB.corpus_sha("hy3d_images", [dict(i) for i in items])
    assert a != MB.corpus_sha("hy3d_images", items[::-1])  # the item order IS the shard order
    assert a != MB.corpus_sha("ace_clips", items)


def test_image_dir_corpus_reads_captions_orders_by_content_and_excludes(tmp_path) -> None:
    d = tmp_path / "batch1"
    d.mkdir()
    for name in ("a_kettle", "b_robot", "c_turtle"):
        (d / f"{name}.png").write_bytes(name.encode() * 50)
    (d / "parent_images_receipt.json").write_text(json.dumps({"images": [{"file": "b_robot.png", "prompt": "a toy robot"}]}))
    excl_sha = MB.sha256_file(d / "c_turtle.png")
    items, counts = MB.image_dir_items([str(d)], names=None, salt="hy3d", held_mod=2, exclude={excl_sha})
    assert counts == {"files": 3, "excluded": 1, "name_filtered": 0}
    by_id = {i["id"]: i for i in items}
    assert by_id["batch1/b_robot"]["caption"] == "a toy robot" and by_id["batch1/a_kettle"]["caption"] == ""
    ordered = MB.order_and_limit(items, salt="hy3d", limit=None)
    assert ordered == MB.order_and_limit(items[::-1], salt="hy3d", limit=None)
    assert len(MB.order_and_limit(items, salt="hy3d", limit=1)) == 1
    picked, _ = MB.image_dir_items([str(d)], names=["b_robot"], salt="hy3d", held_mod=2, exclude=set())
    assert [i["id"] for i in picked] == ["batch1/b_robot"]
    out = tmp_path / "corpus.json"
    body = MB.write_corpus(out, "hy3d_images", ordered, source={}, screen={"applied": True}, deviations=[])
    back = MB.read_corpus(out)
    assert back["corpus_sha256"] == body["corpus_sha256"]
    raw = json.loads(out.read_text())
    raw["items"][0]["seed"] += 1
    out.write_text(json.dumps(raw))
    with pytest.raises(MB.Refusal):
        MB.read_corpus(out)


def test_corpus_refuses_duplicates(tmp_path) -> None:
    items = _items(2)
    items[1]["sha256"] = items[0]["sha256"]
    with pytest.raises(MB.Refusal):
        MB.write_corpus(tmp_path / "c.json", "x", items, source={}, screen={}, deviations=[])


def test_split_caption_lyrics() -> None:
    cap, lyr = MB.split_caption_lyrics("user: [verse] In the quiet of the night [chorus] Swing me high")
    assert cap == "A song." and lyr.startswith("[verse]") and "[chorus]" in lyr
    cap, lyr = MB.split_caption_lyrics("upbeat jazz, brass [Verse 1] hello")
    assert cap == "upbeat jazz, brass" and lyr == "[Verse 1] hello"
    cap, lyr = MB.split_caption_lyrics("Generate an instrumental downtempo track.")
    assert cap == "Generate an instrumental downtempo track." and lyr == ""


def test_prompt_items_are_unique_and_seeded() -> None:
    clips = [{"id": "r1", "sha256": "a" * 64, "caption": "jazz", "lyrics": ""},
             {"id": "r2", "sha256": "b" * 64, "caption": "jazz", "lyrics": ""},  # same prompt -> deduplicated
             {"id": "r3", "sha256": "c" * 64, "caption": "rock", "lyrics": "[verse] x"}]
    items = MB.prompt_items(clips, seeds_per_prompt=3, seconds=60.0, held_mod=8)
    assert len(items) == 6 and len({i["sha256"] for i in items}) == 6 and len({i["seed"] for i in items}) == 6
    assert all(i["seconds"] == 60.0 for i in items)


def test_rank_items_partition() -> None:
    corpus = {"items": _items(10)}
    parts = [MB.rank_items(corpus, r, 3, None) for r in range(3)]
    assert sorted(sum(parts, [])) == list(range(10)) and not (set(parts[0]) & set(parts[1]))
    assert MB.rank_items(corpus, 1, 3, 2) == [1, 4]


# ---------------------------------------------------------------- shard store / seal


@pytest.fixture()
def store_env(tmp_path, monkeypatch):
    teacher_file = tmp_path / "teacher.bin"
    teacher_file.write_bytes(b"weights" * 100)
    monkeypatch.setattr(MB, "teacher_files", lambda teacher: [teacher_file])
    items = _items(2)
    cpath = tmp_path / "corpus.json"
    MB.write_corpus(cpath, "hy3d_images", items, source={}, screen={"applied": False}, deviations=[])
    corpus = MB.read_corpus(cpath)
    store = MB.ShardStore(tmp_path / "cache", "hy3d_trajectories", corpus, MB.run_config("hy3d_trajectories", _hy3d_args()))
    store.claim()
    MB.record_integrity(store, "before", 0, "full")
    return store, corpus, teacher_file


def _write_two(store) -> None:
    for i in range(2):
        store.write(i, {"x_t": torch.randn(3, 4, 2), "t_native": torch.linspace(0, 1, 3)}, {"meta_summary": {"k": i}})


def test_shard_store_write_resume_seal_verify(store_env) -> None:
    store, corpus, _ = store_env
    assert not store.done(0)
    _write_two(store)
    assert store.done(0) and store.done(1)
    manifest = MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")
    assert manifest["shard_read_order"] == ["shard-00000.safetensors", "shard-00001.safetensors"]
    assert manifest["teacher_integrity"]["equal"] and manifest["finite"] and manifest["corpus"]["sha256"] == corpus["corpus_sha256"]
    assert manifest["run_config"] == store.config and manifest["config_sha256"] == store.dir.name == store.config_sha
    assert manifest["shard_tool_versions"] == {MB.TOOL_VERSION: 2}
    assert MB.verify(store.dir)["verified"]


def test_tampered_shard_refused(store_env) -> None:
    store, corpus, _ = store_env
    _write_two(store)
    MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")
    data, _ = store.paths(1)
    save_file({"x_t": torch.zeros(3, 4, 2), "t_native": torch.zeros(3)}, str(data))
    assert not store.done(1)
    with pytest.raises(MB.Refusal):
        MB.verify(store.dir)


def test_missing_shard_and_changed_teacher_refused(store_env) -> None:
    store, corpus, teacher_file = store_env
    store.write(0, {"x_t": torch.ones(2)}, {})
    with pytest.raises(MB.Refusal, match="missing"):
        MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")
    store.write(1, {"x_t": torch.ones(2)}, {})
    teacher_file.write_bytes(b"changed" * 100)
    with pytest.raises(MB.Refusal, match="changed"):
        MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")


def test_seal_ignores_before_records_over_another_file_set(store_env, tmp_path, monkeypatch) -> None:
    store, corpus, teacher_file = store_env
    other = tmp_path / "other.bin"
    other.write_bytes(b"x" * 10)
    # an older worker hashed a different file list: superseded, not compared
    monkeypatch.setattr(MB, "teacher_files", lambda teacher: [teacher_file, other])
    MB.record_integrity(store, "before", 1, "full")
    monkeypatch.setattr(MB, "teacher_files", lambda teacher: [teacher_file])
    _write_two(store)
    manifest = MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")
    assert manifest["teacher_integrity"]["equal"] and len(manifest["teacher_integrity"]["superseded_before_records"]) == 1
    # the same file set with a different digest still refuses
    teacher_file.write_bytes(b"tampered")
    with pytest.raises(MB.Refusal, match="changed"):
        MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")


def test_nonfinite_tensor_refused(store_env) -> None:
    store, _, _ = store_env
    bad = torch.ones(4)
    bad[2] = float("nan")
    with pytest.raises(MB.Refusal, match="non-finite"):
        store.write(0, {"v_guided": bad}, {})
    assert not store.paths(0)[0].exists()


# ---------------------------------------------------------------- 1.0.5: run config in the address, resume/seal compat


def test_run_config_changes_the_address_and_ignores_check_only_args(tmp_path) -> None:
    cpath = tmp_path / "ace.json"
    MB.write_corpus(cpath, "ace_clips", _items(2), source={}, screen={"applied": False}, deviations=[])
    root = str(tmp_path / "root")
    base = ["ace-dit-run", "--corpus", str(cpath), "--cache-root", root]
    d1 = MB.worker_cache_dir(["--", *base])
    assert d1 == MB.worker_cache_dir(base + ["--cfg", "1.0", "--rank", "2", "--world", "3", "--limit", "5", "--integrity", "stamp"])
    for changed in (["--cfg", "2"], ["--clip-seconds", "20"], ["--sigmas-per-clip", "2"], ["--vae-device", "cuda:0"], ["--allow-tiled-vae"]):
        assert MB.worker_cache_dir(base + changed) != d1, changed
    assert MB.worker_cache_dir(base + ["--vae-device", "cuda:1"]) == MB.worker_cache_dir(base + ["--vae-device", "cuda:0"])
    assert d1.parent.name == MB.read_corpus(cpath)["corpus_sha256"] and d1.parent.parent.name == "ace_dit_velocity"
    # the x0 stage is keyed on the x0 arguments only: --cfg 2 reuses the cfg-1 encodes
    s1 = MB.worker_cache_dir(["ace-encode", "--corpus", str(cpath), "--cache-root", root])
    assert s1.name.startswith("x0stage-") and s1.parent == d1.parent
    assert MB.X0Stage(root, MB.read_corpus(cpath), MB.ace_x0_knobs(MB.build_parser().parse_args(base + ["--cfg", "2"]))).dir == s1
    # HY3D: guidance / steps / octree / chunks change the address; replay cadence and the reference arm do not
    hy = MB.run_config("hy3d_trajectories", _hy3d_args())
    assert MB.run_config("hy3d_trajectories", _hy3d_args(replay_every=2, reference_first=0)) == hy
    for kw in ({"guidance": 3.0}, {"steps": 20}, {"octree_resolution": 384}, {"num_chunks": 4000}):
        assert MB.run_config("hy3d_trajectories", _hy3d_args(**kw)) != hy, kw
    parse = MB.build_parser().parse_args
    q = parse(["qvl-run", "--corpus", "c"])
    assert MB.run_config("qvl32_vit_l13", q) != MB.run_config("qvl32_vit_l13", parse(["qvl-run", "--corpus", "c", "--block", "12"]))
    assert MB.run_config("qvl32_vit_l13", q) == MB.run_config("qvl32_vit_l13", parse(["qvl-run", "--corpus", "c", "--reference-s7", ""]))
    lm = parse(["ace-lm-run", "--corpus", "c"])
    assert MB.run_config("ace_lm_codes", lm) == MB.run_config("ace_lm_codes", parse(["ace-lm-run", "--corpus", "c", "--no-parity-first"]))
    assert MB.run_config("ace_lm_codes", lm) != MB.run_config("ace_lm_codes", parse(["ace-lm-run", "--corpus", "c", "--topk", "32"]))


def _rewrite_json(path: Path, **changes) -> None:
    body = json.loads(path.read_text())
    body.update(changes)
    path.write_text(json.dumps(body))


def test_resume_and_seal_refuse_other_config_or_version(store_env) -> None:
    store, corpus, _ = store_env
    _write_two(store)
    _, j1 = store.paths(1)
    _rewrite_json(j1, config_sha256="f" * 64)  # a shard of another run config in this dir
    assert store.done(0) and not store.done(1)
    with pytest.raises(MB.Refusal, match="does not belong in this cache: config"):
        MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")
    _rewrite_json(j1, config_sha256=store.config_sha, tool_version="1.0.4")  # an older tool's shard
    assert not store.done(1)
    with pytest.raises(MB.Refusal, match="not in COMPAT"):
        MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")
    _rewrite_json(j1, tool_version=MB.TOOL_VERSION)
    MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")
    assert MB.verify(store.dir)["verified"]
    _rewrite_json(j1, config_sha256="e" * 64)  # changed after the seal
    with pytest.raises(MB.Refusal, match="not this cache's"):
        MB.verify(store.dir)


def test_claim_refuses_a_different_config_in_the_dir(store_env) -> None:
    store, corpus, _ = store_env
    store.claim()  # idempotent for the same config
    (store.dir / "cache_config.json").write_text(json.dumps({**store.config, "knobs": {"guidance": 2.0}}))
    with pytest.raises(MB.Refusal, match="another run config"):
        store.claim()
    with pytest.raises(MB.Refusal, match="addressed"):
        MB.claim_config(store.dir.parent / "not-a-sha", "cache_config.json", store.config)


def test_verify_refuses_a_pre_1_0_5_manifest(store_env) -> None:
    store, corpus, _ = store_env
    _write_two(store)
    MB.seal(store.dir, corpus, "hy3d_trajectories", mode="full")
    mpath = store.dir / "manifest.json"
    body = json.loads(mpath.read_text())
    del body["run_config"]
    mpath.write_text(json.dumps(body))
    (store.dir / "manifest.json.sha256").write_text(MB.sha256_file(mpath) + "\n")
    with pytest.raises(MB.Refusal, match="no run_config"):
        MB.verify(store.dir)
    legacy = store.dir.parent / "legacy"
    legacy.mkdir()
    with pytest.raises(MB.Refusal, match="cache_config.json missing"):
        MB.ShardStore.from_dir(legacy, "hy3d_trajectories", corpus)


def test_x0_stage_round_trip_tamper_and_item_check(tmp_path) -> None:
    cpath = tmp_path / "ace.json"
    MB.write_corpus(cpath, "ace_clips", _items(2), source={}, screen={"applied": False}, deviations=[])
    corpus = MB.read_corpus(cpath)
    args = MB.build_parser().parse_args(["ace-encode", "--corpus", str(cpath)])
    stage = MB.X0Stage(tmp_path / "root", corpus, MB.ace_x0_knobs(args))
    stage.claim()
    assert stage.load(0) is None
    x0 = torch.randn(1, 64, 750)
    stage.save(0, x0, {"encode_seconds": 1.5, "tiled_fallback": False})
    got = stage.load(0)
    assert got is not None and torch.equal(got[0], x0) and got[1]["tiled_fallback"] is False
    data, jpath = stage.paths(0)
    save_file({"x0": torch.zeros(1, 64, 750)}, str(data))  # tampered: re-encoded, never trusted
    assert stage.load(0) is None
    stage.save(0, x0, {"encode_seconds": 1.5})
    assert stage.load(0) is not None
    _rewrite_json(jpath, item_sha256="a" * 64)  # staged for another item (item 0's synthetic sha is "0" * 64)
    assert stage.load(0) is None
    bad = torch.ones(1, 2, 3)
    bad[0, 1, 1] = float("nan")
    with pytest.raises(MB.Refusal, match="non-finite"):
        stage.save(1, bad, {})
    other_args = MB.build_parser().parse_args(["ace-encode", "--corpus", str(cpath), "--clip-seconds", "10"])
    assert MB.X0Stage(tmp_path / "root", corpus, MB.ace_x0_knobs(other_args)).dir != stage.dir


def test_cpu_only_launch_path() -> None:
    argv = MB.docker_argv(card=None, name="t", tool_args=["ace-encode", "--corpus", "c"], uid=1, gid=1, threads=8)
    assert "--gpus" not in argv and "MEDIA_B_CPU_ONLY=1" in argv and "NVIDIA_VISIBLE_DEVICES=void" in argv and "CUDA_VISIBLE_DEVICES=" in argv
    i = argv.index(MB.RUNTIME_IMAGE)
    assert argv[i + 1: i + 3] == ["-n", "15"] and "OMP_NUM_THREADS=8" in argv
    with pytest.raises(MB.Refusal, match="CPU-only container"):
        MB.docker_argv(card=None, name="t", tool_args=["ace-dit-run"], uid=1, gid=1)
    parser = MB.build_parser()
    with pytest.raises(MB.Refusal, match="is CPU-only"):
        MB.run_launch(parser.parse_args(["launch", "--card", "3", "--name", "t", "--log", "/tmp/x", "--", "ace-encode", "--corpus", "c"]))
    with pytest.raises(MB.Refusal, match="takes no --card"):
        MB.run_launch(parser.parse_args(["launch", "--cpu-only", "--card", "3", "--name", "t", "--log", "/tmp/x", "--", "ace-encode", "--corpus", "c"]))
    with pytest.raises(MB.Refusal, match="needs --card"):
        MB.run_launch(parser.parse_args(["launch", "--name", "t", "--log", "/tmp/x", "--", "hy3d-run", "--corpus", "c"]))
    assert parser.parse_args(["ace-encode", "--corpus", "c"]).vae_device == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["ace-encode", "--corpus", "c", "--vae-device", "cuda:0"])


def test_cpu_only_comfy_import_blocks_triton_and_sets_cpu(tmp_path, monkeypatch) -> None:
    """Unit check of the guard only (a stub comfy tree); the real proof is the CPU-only container run."""

    root = tmp_path / "ComfyUI"
    (root / "comfy/text_encoders").mkdir(parents=True)
    (root / "comfy/__init__.py").write_text("")
    (root / "comfy/text_encoders/__init__.py").write_text("")
    (root / "comfy/text_encoders/ace15.py").write_text("")
    (root / "comfy/sd.py").write_text("")
    (root / "comfy/utils.py").write_text("")
    (root / "comfy/cli_args.py").write_text("import types\nargs = types.SimpleNamespace(cpu=False)\n")
    (root / "comfy/model_management.py").write_text(
        "import comfy.cli_args\nCPU = comfy.cli_args.args.cpu\ntry:\n    import triton\n    TRITON = 'imported'\n"
        "except ImportError:\n    TRITON = 'importerror'\n")
    saved = {k: v for k, v in sys.modules.items() if k == "triton" or k == "comfy" or k.startswith("comfy.")}
    for k in saved:
        del sys.modules[k]
    monkeypatch.setattr(MB, "COMFY_ROOT", root)
    monkeypatch.setenv("MEDIA_B_CPU_ONLY", "1")
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        c = MB.comfy_modules()
        assert c.mm.CPU is True and c.mm.TRITON == "importerror"
    finally:
        for k in [k for k in sys.modules if k == "triton" or k == "comfy" or k.startswith("comfy.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def test_ace_posterior_sample_is_seeded_per_item_and_leaves_rng_alone() -> None:
    """encode_audio seeds the Oobleck posterior sample (a fake VAE that samples like comfy's vae_sample)."""

    class FakeVae:
        vae_dtype = torch.float32

        def encode(self, x):
            mean = x.mean(dim=1, keepdim=True)[..., :8]
            return torch.randn_like(mean) * 0.01 + mean

    teacher = object.__new__(MB.AceTeacher)
    teacher.vae_device = torch.device("cpu")
    teacher.vae = FakeVae()
    wav = torch.randn(2, 480)
    state = torch.get_rng_state()
    a, meta = teacher.encode_audio(wav, seed=11)
    b, _ = teacher.encode_audio(wav, seed=11)
    c, _ = teacher.encode_audio(wav, seed=12)
    assert torch.equal(a, b) and not torch.equal(a, c) and meta["posterior_seed"] == 11
    assert torch.equal(torch.get_rng_state(), state)  # the global RNG is untouched (fork_rng)
    knobs = MB.ace_x0_knobs(MB.build_parser().parse_args(["ace-encode", "--corpus", "c"]))
    assert MB.ACE_VAE_POSTERIOR_SALT in knobs["posterior_sample"]
    assert MB.COMPAT["ace_dit_velocity"][0] == "1.0.7" and "1.0.5" not in MB.COMPAT["ace_dit_velocity"]


def test_s7_forward_slice() -> None:
    forwards = [{"index": 0, "image": "/a.png", "side": 448, "rows": 784}, {"index": 1, "image": "/a.png", "side": 512, "rows": 1024},
                {"index": 2, "image": "/b.png", "side": 448, "rows": 784}, {"index": 3, "image": "/b.png", "side": 512, "rows": 1024}]
    sha = {"/a.png": "aa", "/b.png": "bb"}
    assert MB.s7_forward_slice(forwards, sha, "bb", 512, 512) == (3, 784 + 1024 + 784, 1024)
    assert MB.s7_forward_slice(forwards, sha, "aa", 448, 448) == (0, 0, 784)
    assert MB.s7_forward_slice(forwards, sha, "aa", 512, 448) is None  # not square: s7 resized to side x side
    assert MB.s7_forward_slice(forwards, sha, "cc", 512, 512) is None
    assert MB.s7_forward_slice(forwards, sha, "aa", 384, 384) is None


def test_qvl_reference_gate_passes_tf32_drift_and_refuses_transcription_errors() -> None:
    torch.manual_seed(0)
    ref = torch.randn(1024, 1152) * 0.4
    ref[7, 3] = 300.0  # a massive-activation row, as in the real block-13 input
    tf32_like = ref * (1 + 3e-3 * torch.randn_like(ref))  # ~TF32 relative drift
    tf32_like[7, 3] = 300.0 * (1 + 3e-3)  # 0.9 absolute on the massive row
    st = MB.ref_row_stats(tf32_like, ref)
    assert st["rel_fro"] <= MB.QVL_REF_REL_FRO and st["max_abs_diff"] > 0.3 and not st["bitwise_equal"]  # 1.0.5 max_abs gate: refused
    assert MB.ref_row_stats(ref.clone(), ref)["bitwise_equal"] and MB.ref_row_stats(ref.clone(), ref)["rel_fro"] == 0.0
    permuted = ref[torch.randperm(1024)]  # wrong patch order
    assert MB.ref_row_stats(permuted, ref)["rel_fro"] > 10 * MB.QVL_REF_REL_FRO
    renorm = ref * 1.12 + 0.05  # wrong normalisation constants, propagated
    assert MB.ref_row_stats(renorm, ref)["rel_fro"] > MB.QVL_REF_REL_FRO


def test_check_summary_rolls_up_reference_and_determinism() -> None:
    shards = [{"meta": {"ref_max_abs_diff": 0.0, "ref_bitwise": True, "replay_max_abs_diff": 0.0, "replay_bitwise": True}},
              {"meta": {"replay_max_abs_diff": 2e-7, "replay_bitwise": False}}, {"meta": {}}]
    out = MB.check_summary(shards)
    assert out == {"reference_arms": 1, "reference_bitwise_equal": 1, "reference_max_abs_diff": 0.0, "determinism_replays": 2,
                   "determinism_bitwise_equal": 1, "determinism_max_abs_diff": 2e-7}
    assert MB.check_summary([])["reference_max_abs_diff"] is None


# ---------------------------------------------------------------- math pieces


def test_receipts_survive_nonfinite_floats(tmp_path) -> None:
    path = tmp_path / "r.json"
    MB.write_json_atomic(path, {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": -float("inf")}}, sidecar_sha=True)
    body = json.loads(path.read_text())
    assert body == {"a": "nan", "b": [1.0, "inf"], "c": {"d": "-inf"}}
    assert (tmp_path / "r.json.sha256").read_text().strip() == MB.sha256_file(path)


def test_require_finite_names_the_tensor() -> None:
    MB.require_finite("ok", torch.ones(3))
    bad = torch.ones(3)
    bad[1] = float("inf")
    with pytest.raises(MB.Refusal, match="non-finite x0"):
        MB.require_finite("x0", bad)


def test_cfg_combine() -> None:
    c, u = torch.randn(5), torch.randn(5)
    assert torch.equal(MB.cfg_combine(c, u, 1.0), c)
    assert torch.allclose(MB.cfg_combine(c, u, 5.0), u + 5.0 * (c - u))


def _upstream_hy3d_sigmas(steps: int) -> torch.Tensor:
    """Independent restatement of hy3dshape FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=linspace(0,1)),
    shift 1, not dynamic: sigmas = shift*s/(1+(shift-1)*s) = s, then a trailing 1.0 is appended."""

    s = torch.linspace(0, 1, steps, dtype=torch.float64).to(torch.float32)
    return torch.cat([s, torch.ones(1)])


def test_hy3d_time_convention_and_euler_chain() -> None:
    assert MB.hy3d_noise_level(0.0) == 1.0 and MB.hy3d_noise_level(1.0) == 0.0
    sig = _upstream_hy3d_sigmas(30)
    x = [torch.randn(16, 4)]
    vs = []
    for i in range(30):
        v = torch.randn(16, 4)
        vs.append(v)
        x.append((x[-1].float() + (sig[i + 1] - sig[i]) * v).to(v.dtype))  # upstream step: sample + (sigma_next - sigma) * output
    xt, vt = torch.stack(x[:30]), torch.stack(vs)
    assert MB.euler_chain_residual(xt, vt, sig) < 1e-6
    broken = xt.clone()
    broken[7] += 0.5
    assert MB.euler_chain_residual(broken, vt, sig) > 1e-3
    assert float(sig[-2]) == 1.0 and float(sig[-1]) == 1.0  # the last of 30 steps has dt = 0 (upstream quirk, recorded)


def _comfy_simple_sigmas(steps: int, shift: float) -> list[float]:
    """Independent restatement in torch of comfy ModelSamplingDiscreteFlow(shift, multiplier=1.0) + simple_scheduler."""

    t = (torch.arange(1, 1001, 1) / 1000) * 1.0
    sigmas = shift * (t / 1.0) / (1 + (shift - 1) * (t / 1.0))
    ss = len(sigmas) / steps
    return [float(sigmas[-(1 + int(x * ss))]) for x in range(steps)]


def test_ace_schedule_matches_comfy_simple() -> None:
    ours = MB.ace_schedule()
    ref = _comfy_simple_sigmas(8, 3.0)
    assert len(ours) == 8 and max(abs(a - b) for a, b in zip(ours, ref)) < 1e-6
    assert abs(ours[0] - 1.0) < 1e-7 and all(a > b for a, b in zip(ours, ours[1:]))
    covered = set(MB.ace_clip_sigmas(0, ours)) | set(MB.ace_clip_sigmas(1, ours))
    assert covered == set(ours)
    assert len(set(MB.ace_clip_sigmas(5, ours))) == 4


def test_band_topk_over_audio_codes() -> None:
    v = 217204
    logits = torch.randn(v) * 0.1
    logits[100] = 50.0  # a huge text-token logit OUTSIDE the band must not appear in the ids
    planted = [MB.ACE_AUDIO_START + 7, MB.ACE_AUDIO_START + 63999, MB.ACE_AUDIO_START + 1234]
    for k, idx in enumerate(planted):
        logits[idx] = 10.0 - k
    out = MB.band_topk(logits, k=64)
    assert out["ids"].dtype == torch.int32 and out["ids"].shape == (64,)
    assert out["ids"][:3].tolist() == [7, 63999, 1234]
    assert int(out["ids"].min()) >= 0 and int(out["ids"].max()) < MB.ACE_CODES
    lse = torch.logsumexp(logits, 0)
    assert torch.allclose(out["logprobs"][0], logits[planted[0]] - lse)
    assert float(out["lse_band"]) < float(out["lse_full"])  # the text logit at 100 holds most of the mass
    assert torch.allclose(out["eos_logprob"], logits[MB.ACE_EOS] - lse)


def test_smart_resize() -> None:
    for hw in ((512, 512), (1080, 1920), (300, 200), (4000, 3000), (40, 40)):
        h, w = MB.smart_resize(*hw)
        assert h % 32 == 0 and w % 32 == 0 and h * w <= 512 * 512 and h * w >= 32 * 32
    assert MB.smart_resize(512, 512) == (512, 512)


# ---------------------------------------------------------------- P8 header


def test_header_summary_detects_lm_head_and_layers(tmp_path) -> None:
    full = {"model.language_model.layers.0.mlp.w": torch.zeros(2, 2), "model.language_model.layers.1.mlp.w": torch.zeros(2, 2),
            "model.language_model.embed_tokens.weight": torch.zeros(10, 2), "lm_head.weight": torch.zeros(10, 2),
            "visual.blocks.0.attn.qkv.weight": torch.zeros(6, 2), "visual.blocks.1.attn.qkv.weight": torch.zeros(6, 2),
            "visual.blocks.2.attn.qkv.weight": torch.zeros(6, 2)}
    path = tmp_path / "full.safetensors"
    save_file(full, str(path))
    s = MB.header_summary(MB.read_safetensors_header(path))
    assert s["has_lm_head"] and s["decoder_layers"] == 2 and s["vision_blocks"] == 3 and s["tensors"] == 7
    assert s["parameters"] == 4 + 4 + 20 + 20 + 36
    del full["lm_head.weight"]
    save_file(full, str(path))
    assert not MB.header_summary(MB.read_safetensors_header(path))["has_lm_head"]


# ---------------------------------------------------------------- GPU guard / launch


SMI = """0, GPU-4ead, NVIDIA GeForce RTX 3090, 24, 24576
1, GPU-p401, Tesla P40, 5, 24576
2, GPU-p402, Tesla P40, 2500, 24576
3, GPU-p403, Tesla P40, 5, 24576
4, GPU-2ce6, NVIDIA GeForce RTX 3090, 15, 24576
5, GPU-p405, Tesla P40, 5, 24576
6, GPU-q800, Quadro RTX 8000, 15862, 46080"""


def test_gpu_guard() -> None:
    gpus = MB.parse_nvidia_smi(SMI)
    assert gpus[3]["name"] == "Tesla P40" and gpus[6]["used_mib"] == 15862
    assert MB.check_card(3, gpus)["uuid"] == "GPU-p403"
    for card in (0, 4, 5, 6):
        with pytest.raises(MB.Refusal, match="refused"):
            MB.check_card(card, gpus)
    with pytest.raises(MB.Refusal, match="2000"):
        MB.check_card(2, gpus)
    with pytest.raises(MB.Refusal):
        MB.check_card(7, gpus)


def test_docker_argv_one_card_nice_threads_readonly_models() -> None:
    argv = MB.docker_argv(card=3, name="t", tool_args=["selfcheck"], uid=1000, gid=1000, threads=8)
    assert argv.count("--gpus") == 1 and argv[argv.index("--gpus") + 1] == "device=3"
    assert "all" not in argv
    assert f"{MB.MODELS}:{MB.MODELS}:ro" in argv and f"{MB.HY3D_SOURCE}:{MB.HY3D_SOURCE}:ro" in argv
    i = argv.index(MB.RUNTIME_IMAGE)
    assert argv[i + 1: i + 3] == ["-n", "15"] and argv[argv.index("--entrypoint") + 1] == "/usr/bin/nice"
    assert "OMP_NUM_THREADS=8" in argv and "MEDIA_B_PHYSICAL_GPU=3" in argv and argv[-1] == "selfcheck"
    assert argv[argv.index("--user") + 1] == "1000:1000"
    for bad in ({"card": 6}, {"card": 5}, {"card": 0}, {"card": 3, "threads": 32}):
        kw = {"card": 3, "name": "t", "tool_args": [], "uid": 1, "gid": 1, "threads": 8, **bad}
        with pytest.raises(MB.Refusal):
            MB.docker_argv(**kw)


def test_pascal_argv_single_visible_card() -> None:
    argv, env = MB.pascal_argv(card=2, tool_args=["qvl-run"], threads=16)
    assert env["CUDA_VISIBLE_DEVICES"] == "2" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert argv[:3] == ["nice", "-n", "15"] and env["OMP_NUM_THREADS"] == "16"
    with pytest.raises(MB.Refusal):
        MB.pascal_argv(card=4, tool_args=[])


# ---------------------------------------------------------------- 1.0.8: exFAT claim, Lexar mount, corpus-import


def _no_hardlinks(monkeypatch) -> None:
    def link(src, dst, *a, **kw):
        raise PermissionError(1, "Operation not permitted", str(src))  # what exFAT returns (measured on /mnt/lexar)
    monkeypatch.setattr(MB.os, "link", link)


def test_claim_without_hardlinks_is_first_writer_wins(tmp_path, monkeypatch) -> None:
    _no_hardlinks(monkeypatch)
    cpath = tmp_path / "c.json"
    MB.write_corpus(cpath, "hy3d_images", _items(2), source={}, screen={"applied": False}, deviations=[])
    corpus = MB.read_corpus(cpath)
    store = MB.ShardStore(tmp_path / "cache", "hy3d_trajectories", corpus, MB.run_config("hy3d_trajectories", _hy3d_args()))
    store.claim()
    store.claim()  # the second worker of the same config: accepted
    assert MB.read_json_settled(store.dir / "cache_config.json") == store.config
    assert not list(store.dir.glob("*.tmp*"))
    other = MB.ShardStore(tmp_path / "cache", "hy3d_trajectories", corpus, MB.run_config("hy3d_trajectories", _hy3d_args(guidance=3.0)))
    assert other.dir != store.dir
    (other.dir).mkdir(parents=True)
    (other.dir / "cache_config.json").write_text(json.dumps(store.config))  # a dir holding another config
    with pytest.raises(MB.Refusal, match="another run config"):
        other.claim()
    # an errno that is not "no hard links" is not swallowed
    def link_eio(src, dst, *a, **kw):
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(MB.os, "link", link_eio)
    fresh = MB.ShardStore(tmp_path / "cache2", "hy3d_trajectories", corpus, MB.run_config("hy3d_trajectories", _hy3d_args()))
    with pytest.raises(MB.Refusal, match="claim failed"):
        fresh.claim()


def test_create_exclusive_and_settled_read(tmp_path) -> None:
    import threading

    path = tmp_path / "x.json"
    assert MB.create_exclusive(path, b'{"a": 1}\n') is True
    assert MB.create_exclusive(path, b'{"a": 2}\n') is False
    assert MB.read_json_settled(path) == {"a": 1}
    partial = tmp_path / "p.json"
    partial.write_text('{"a"')  # a writer caught mid-write
    threading.Timer(0.3, lambda: partial.write_text('{"a": 3}')).start()
    assert MB.read_json_settled(partial, tries=40, pause=0.05) == {"a": 3}
    stuck = tmp_path / "s.json"
    stuck.write_text("")
    with pytest.raises(MB.Refusal, match="not a complete json"):
        MB.read_json_settled(stuck, tries=3, pause=0.01)


def test_worker_write_mounts_lexar_root_only(tmp_path, monkeypatch) -> None:
    allowed = tmp_path / "lexar/omni-campaign/teacher_cache"
    monkeypatch.setattr(MB, "TEACHER_CACHE_WRITE_ROOTS", (allowed,))
    monkeypatch.setattr(MB, "OC", tmp_path / "oc")
    root = allowed / "hy3d"
    mounts = MB.worker_write_mounts(["--", "hy3d-run", "--corpus", "c", "--cache-root", str(root)])
    assert mounts == [f"{root}:{root}:rw"] and root.is_dir()
    assert MB.worker_write_mounts(["hy3d-run", "--corpus", "c", "--cache-root", str(tmp_path / "oc/distill/x")]) == []
    assert MB.worker_write_mounts(["selfcheck"]) == []
    with pytest.raises(MB.Refusal, match="outside"):
        MB.worker_write_mounts(["qvl-run", "--corpus", "c", "--cache-root", str(tmp_path / "lexar/omni-campaign/omni1")])
    argv = MB.docker_argv(card=1, name="t", tool_args=["hy3d-run"], uid=1, gid=1, threads=8, extra_mounts=mounts)
    vols = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert "/mnt/lexar/omni-campaign:/mnt/lexar/omni-campaign:ro" in vols and f"{root}:{root}:rw" in vols


def _sha(data: bytes) -> str:
    return MB.sha256_bytes(data)


def _distill_set(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    d = tmp_path / "media" / name
    d.mkdir(parents=True)
    items = d / "items.jsonl"
    items.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "manifest.json").write_text(json.dumps({"schema": MB.DISTILL_SET_SCHEMA + name, "identity": {"tool": "distill_corpus_build"},
                                                 "items_file": {"path": str(items), "sha256": MB.sha256_file(items), "items": len(rows)},
                                                 "excluded": {"eval_split_and_screened_rows": "never in pool"}}))
    return d


def _image_rows(tmp_path: Path, n: int) -> list[dict]:
    rows = []
    for k in range(n):
        data = f"img{k}".encode() * 40
        p = tmp_path / f"store/{k}.png"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        rows.append({"rid": f"000:{k}:1", "dataset_id": "ds", "caption": f"object {k}", "tier": "icon", "route": "image",
                     "image": {"path": "/nonexistent/original.png", "stored_path": str(p), "sha256": _sha(data)}})
    return rows


def test_corpus_import_images(tmp_path) -> None:
    rows = _image_rows(tmp_path, 5)
    hy = _distill_set(tmp_path, "hy3d", rows)
    out = tmp_path / "hy3d.json"
    MB.main(["corpus-import", "--set-dir", str(hy), "--kind", "hy3d_images", "--out", str(out)])
    body = MB.read_corpus(out)
    assert body["kind"] == "hy3d_images" and body["n_items"] == 5 and body["contamination_screen"]["applied"] is True
    assert {i["sha256"] for i in body["items"]} == {r["image"]["sha256"] for r in rows}
    assert all(Path(i["path"]).is_file() and i["caption"].startswith("object") for i in body["items"])
    assert body["source"]["distill_set"]["items_sha256"] == MB.sha256_file(hy / "items.jsonl")
    assert any("background removal" in d for d in body["deviations"])
    # same rows, other order in the set file -> the same corpus (content-keyed order)
    hy2 = _distill_set(tmp_path / "b", "hy3d", rows[::-1])
    out2 = tmp_path / "hy3d2.json"
    MB.main(["corpus-import", "--set-dir", str(hy2), "--kind", "hy3d_images", "--out", str(out2)])
    assert MB.read_corpus(out2)["corpus_sha256"] == body["corpus_sha256"]
    vit = _distill_set(tmp_path / "c", "vit", rows)
    out3 = tmp_path / "qvl.json"
    assert MB.main(["corpus-import", "--set-dir", str(vit), "--kind", "qvl_images", "--limit", "3", "--out", str(out3)]) == 0
    q = MB.read_corpus(out3)
    assert q["kind"] == "qvl_images" and q["n_items"] == 3 and all(i["route"] == "image" for i in q["items"])
    # a set of another kind is refused
    assert MB.main(["corpus-import", "--set-dir", str(vit), "--kind", "hy3d_images", "--out", str(tmp_path / "no.json")]) == 3


def test_corpus_import_refuses_changed_inputs(tmp_path) -> None:
    rows = _image_rows(tmp_path, 3)
    hy = _distill_set(tmp_path, "hy3d", rows)
    (hy / "items.jsonl").write_text((hy / "items.jsonl").read_text() + "\n")  # items file no longer the manifest's
    with pytest.raises(MB.Refusal, match="sha256"):
        MB.read_distill_set(hy)
    hy2 = _distill_set(tmp_path / "b", "hy3d", rows)
    Path(rows[1]["image"]["stored_path"]).write_bytes(b"changed")  # an image changed after the corpus lane hashed it
    with pytest.raises(MB.Refusal, match="recorded"):
        MB.run_corpus_import(MB.build_parser().parse_args(["corpus-import", "--set-dir", str(hy2), "--kind", "hy3d_images",
                                                            "--out", str(tmp_path / "x.json")]))


def test_corpus_import_ace_clips_and_prompts(tmp_path) -> None:
    rows = []
    for k in range(4):
        data = f"flac{k}".encode() * 30
        p = tmp_path / f"clips/ace-{k:04d}.flac"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        caption = "upbeat jazz [verse] la la" if k % 2 else "an instrumental ambient piece"
        if k == 3:
            caption = "an instrumental ambient piece"  # same prompt as clip 0 / 2 -> deduplicated for the LM
        rows.append({"rid": f"001:0:{k}", "dataset_id": "ds", "caption": caption, "seconds": 12.52, "tier": "t",
                     "clip": {"path": str(p), "sha256": _sha(data)}})
    # a second registry row cutting the same window of the same source: byte-identical clip under another name
    dup = tmp_path / "clips/ace-0099.flac"
    dup.write_bytes(Path(rows[1]["clip"]["path"]).read_bytes())
    rows.append({**rows[1], "rid": "000:9:9", "clip": {"path": str(dup), "sha256": rows[1]["clip"]["sha256"]}})
    ace = _distill_set(tmp_path, "ace", rows)
    out = tmp_path / "clips.json"
    assert MB.main(["corpus-import", "--set-dir", str(ace), "--kind", "ace_clips", "--out", str(out)]) == 0
    clips = MB.read_corpus(out)
    assert clips["kind"] == "ace_clips" and clips["n_items"] == 4
    assert clips["source"]["duplicate_content_rows"] == [{"rid": "000:9:9", "sha256": rows[1]["clip"]["sha256"], "kept_rid": "001:0:1"}]
    assert any("byte-identical" in d for d in clips["deviations"])
    jazz = next(i for i in clips["items"] if i["lyrics"])
    assert jazz["caption"] == "upbeat jazz" and jazz["lyrics"] == "[verse] la la" and jazz["clip_seconds"] == 12.52
    assert any("--clip-seconds" in d and "12.52" in d for d in clips["deviations"])
    outp = tmp_path / "prompts.json"
    assert MB.main(["corpus-import", "--set-dir", str(ace), "--kind", "ace_prompts", "--seeds-per-prompt", "3", "--seconds", "60",
                    "--out", str(outp)]) == 0
    prompts = MB.read_corpus(outp)
    assert prompts["kind"] == "ace_prompts" and prompts["n_items"] == 2 * 3  # 2 distinct (caption, lyrics) x 3 seeds
    assert all(i["seconds"] == 60.0 for i in prompts["items"])
    outl = tmp_path / "prompts4.json"
    MB.main(["corpus-import", "--set-dir", str(ace), "--kind", "ace_prompts", "--seeds-per-prompt", "3", "--limit", "4", "--out", str(outl)])
    assert MB.read_corpus(outl)["n_items"] == 4


def _sealed(tmp_path, monkeypatch, teacher: str, kind: str, worker: list[str], shards: list[tuple[dict, dict]]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    teacher_file = tmp_path / "teacher.bin"
    teacher_file.write_bytes(b"w" * 64)
    monkeypatch.setattr(MB, "teacher_files", lambda t: [teacher_file])
    cpath = tmp_path / f"{teacher}.json"
    MB.write_corpus(cpath, kind, _items(len(shards)), source={}, screen={"applied": True}, deviations=[])
    corpus = MB.read_corpus(cpath)
    store = MB.ShardStore(tmp_path / "cache", teacher, corpus, MB.run_config(teacher, MB.build_parser().parse_args([*worker, "--corpus", "c"])))
    store.claim()
    MB.record_integrity(store, "before", 0, "full")
    for index, (tensors, meta) in enumerate(shards):
        store.write(index, tensors, meta)
    MB.seal(store.dir, corpus, teacher, mode="full")
    return store.dir


def _hy3d_shard(store_dir_name: str, index: int, faces: int) -> tuple[dict, dict]:
    ply = f"shard-{index:05d}.mesh.ply" if faces else None
    return ({"v_guided": torch.ones(30, 4, 2), "mesh_faces": torch.zeros(faces, 3, dtype=torch.int32)},
            {"mesh": {"faces": faces, "ply": ply, "seconds": 140.0}, "meta_summary": {"faces": faces, "s_per_cfg_forward": 9.2, "held": False}})


def test_check_hy3d_faces_and_ply(tmp_path, monkeypatch) -> None:
    d = _sealed(tmp_path, monkeypatch, "hy3d_trajectories", "hy3d_images", ["hy3d-run"], [_hy3d_shard("", 0, 12), _hy3d_shard("", 1, 0)])
    (d / "shard-00000.mesh.ply").write_bytes(b"ply")
    out = tmp_path / "check.json"
    assert MB.main(["check", "--cache-dir", str(d), "--out", str(out)]) == 3  # shard 1 decoded no mesh
    body = json.loads(out.read_text())
    assert body["verdict"] == "FAIL" and body["stats"]["pairs"] == 60 and body["stats"]["zero_face_items"] == ["i1"]
    assert (tmp_path / "check.json.sha256").read_text().strip() == MB.sha256_file(out)
    d2 = _sealed(tmp_path / "b", monkeypatch, "hy3d_trajectories", "hy3d_images", ["hy3d-run"], [_hy3d_shard("", 0, 5)])
    assert MB.main(["check", "--cache-dir", str(d2), "--out", str(tmp_path / "c2.json")]) == 3  # .ply missing
    (d2 / "shard-00000.mesh.ply").write_bytes(b"ply")
    assert MB.main(["check", "--cache-dir", str(d2), "--out", str(tmp_path / "c3.json")]) == 0
    assert json.loads((tmp_path / "c3.json").read_text())["verdict"] == "PASS"


def test_check_ace_x0_lm_and_qvl(tmp_path, monkeypatch) -> None:
    def ace(x0):
        return ({"x0": x0, "v_guided": torch.ones(4, 2, 313)}, {"meta_summary": {"s_per_guided_forward": 0.2, "tiled_vae": False}})
    d = _sealed(tmp_path / "a", monkeypatch, "ace_dit_velocity", "ace_clips", ["ace-dit-run"], [ace(torch.randn(2, 313)), ace(torch.zeros(2, 313))])
    out = tmp_path / "ace.json"
    assert MB.main(["check", "--cache-dir", str(d), "--out", str(out)]) == 3
    body = json.loads(out.read_text())
    assert body["problems"] == ["all-zero ACE x0: i1"] and body["stats"]["velocity_targets"] == 8 and body["stats"]["latent_frames"] == [313]
    lm = [({"tokens": torch.arange(300, dtype=torch.int32)}, {"eos_hit": False, "mean_band_mass": 0.98, "meta_summary": {"tok_per_s": 11.0}})] * 2
    d = _sealed(tmp_path / "l", monkeypatch, "ace_lm_codes", "ace_prompts", ["ace-lm-run"], lm)
    assert MB.main(["check", "--cache-dir", str(d), "--out", str(tmp_path / "lm.json")]) == 0
    assert json.loads((tmp_path / "lm.json").read_text())["stats"]["code_tokens"] == 600
    qvl = [_qvl_shard(n, rel_fro=6.8e-3) for n in (4, 6)]  # rows + in-cache s7 arms
    d = _sealed(tmp_path / "q", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], qvl)
    assert MB.main(["check", "--cache-dir", str(d), "--out", str(tmp_path / "q.json")]) == 0
    stats = json.loads((tmp_path / "q.json").read_text())["stats"]
    assert stats["rows"] == 10 and stats["determinism_replay_bitwise"] == 2 and "replay_bitwise" not in stats
    assert stats["parent_reference"]["source"] == "in-cache s7 reference arms" and stats["parent_reference"]["arms"] == 2


# ---------------------------------------------------------------- 1.0.9: QVL parent reference in `check`


def _qvl_shard(n: int, *, rel_fro: float | None = None, side: int = 512) -> tuple[dict, dict]:
    meta = {"seconds": 0.3, "checks": {"determinism_replay": {"bitwise_equal": True, "max_abs_diff": 0.0}, "reference_s7_capture": None},
            "meta_summary": {"rows": n, "replay_bitwise": True, "replay_max_abs_diff": 0.0}}
    if rel_fro is not None:
        ref = {"dir": "/s7/qvl32_vit_L13", "json_sha256": "a" * 64, "file_sha256": "b" * 64, "forward": 1, "rows": [0, n], "side": side,
               "bitwise_equal": False, "max_abs_diff": 4.9, "ref_absmax": 300.0, "row_cosine_min": 0.9966}
        if rel_fro == rel_fro:  # NaN marks a 1.0.5-style arm without rel_fro
            ref["rel_fro"] = rel_fro
        meta["checks"]["reference_s7_capture"] = ref
        meta["meta_summary"].update(ref_max_abs_diff=4.9, ref_bitwise=False)
    return {"block_in": torch.ones(n, 3), "block_out": torch.ones(n, 3)}, meta


def _check(d: Path, out: Path, *extra: str) -> tuple[int, dict]:
    rc = MB.main(["check", "--cache-dir", str(d), "--out", str(out), *extra])
    return rc, json.loads(out.read_text())


def test_check_qvl_without_parent_reference_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MB, "QVL_PARENT_REF_ROOT", tmp_path / "ref_root")
    d = _sealed(tmp_path / "p", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], [_qvl_shard(4), _qvl_shard(6)])
    rc, body = _check(d, tmp_path / "p.json")
    assert rc == 3 and body["verdict"] == "FAIL" and body["stats"]["parent_reference"] is None
    assert body["stats"]["determinism_replay_bitwise"] == 2  # a perfect determinism record does not stand in for the parent
    assert any("no QVL parent reference" in p and "not a parent comparison" in p for p in body["problems"])


def test_check_qvl_finds_a_side_reference_of_the_same_config(tmp_path, monkeypatch) -> None:
    side = _sealed(tmp_path / "ref", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"],
                   [_qvl_shard(1024, rel_fro=r) for r in (2.9e-3, 6.8e-3, 4.8e-3)])
    monkeypatch.setattr(MB, "QVL_PARENT_REF_ROOT", tmp_path / "ref" / "cache")
    d = _sealed(tmp_path / "p", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], [_qvl_shard(4), _qvl_shard(6)])
    assert side.name == d.name  # the same run config sha
    rc, body = _check(d, tmp_path / "p.json")
    parent = body["stats"]["parent_reference"]
    assert rc == 0 and body["verdict"] == "PASS" and parent["cache_dir"] == str(side) and parent["arms"] == 3
    assert parent["rel_fro_max"] == 6.8e-3 and parent["sides"] == [512] and parent["config_sha256"] == d.name
    assert parent["manifest_sha256"] == MB.sha256_file(side / "manifest.json") and parent["source"].startswith("side cache")
    # named explicitly: the same answer
    rc, body = _check(d, tmp_path / "p2.json", "--parent-reference", str(side))
    assert rc == 0 and body["stats"]["parent_reference"]["cache_dir"] == str(side)


def test_check_qvl_refuses_a_bad_side_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MB, "QVL_PARENT_REF_ROOT", tmp_path / "nowhere")
    d = _sealed(tmp_path / "p", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], [_qvl_shard(4)])
    failing = _sealed(tmp_path / "f", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], [_qvl_shard(8, rel_fro=0.31)])
    rc, body = _check(d, tmp_path / "a.json", "--parent-reference", str(failing))
    assert rc == 3 and any("rel_fro 3.100e-01 > 0.02" in p for p in body["problems"])
    old = _sealed(tmp_path / "o", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], [_qvl_shard(8, rel_fro=float("nan"))])
    rc, body = _check(d, tmp_path / "b.json", "--parent-reference", str(old))
    assert rc == 3 and any("without a finite rel_fro" in p for p in body["problems"])
    other = _sealed(tmp_path / "c", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run", "--max-pixels", str(448 * 448)],
                    [_qvl_shard(8, rel_fro=3e-3)])
    rc, body = _check(d, tmp_path / "c.json", "--parent-reference", str(other))
    assert rc == 3 and any("run config" in p for p in body["problems"])
    empty = _sealed(tmp_path / "e", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], [_qvl_shard(8)])
    rc, body = _check(d, tmp_path / "e.json", "--parent-reference", str(empty))
    assert rc == 3 and any("no QVL reference arm" in p for p in body["problems"])
    good = _sealed(tmp_path / "g", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], [_qvl_shard(8, rel_fro=3e-3)])
    assert _check(d, tmp_path / "g.json", "--parent-reference", str(good))[0] == 0
    save_file({"block_in": torch.zeros(8, 3), "block_out": torch.ones(8, 3)}, str(good / "shard-00000.safetensors"))  # tampered
    rc, body = _check(d, tmp_path / "t.json", "--parent-reference", str(good))
    assert rc == 3 and any("refused" in p and "sha256" in p for p in body["problems"])
    rc, body = _check(d, tmp_path / "m.json", "--parent-reference", str(tmp_path / "missing"))
    assert rc == 3 and any("not sealed" in p for p in body["problems"])


def test_check_qvl_own_failing_arm_and_other_block(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MB, "QVL_PARENT_REF_ROOT", tmp_path / "nowhere")
    d = _sealed(tmp_path / "a", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run"], [_qvl_shard(4, rel_fro=1e-2), _qvl_shard(4, rel_fro=0.5)])
    rc, body = _check(d, tmp_path / "a.json")
    assert rc == 3 and body["stats"]["parent_reference"]["arms"] == 2 and len(body["problems"]) == 1
    d = _sealed(tmp_path / "b", monkeypatch, "qvl32_vit_l13", "qvl_images", ["qvl-run", "--block", "12"], [_qvl_shard(4)])
    rc, body = _check(d, tmp_path / "b.json")
    assert rc == 3 and any("block-13 input rows only" in p for p in body["problems"])


# ---------------------------------------------------------------- real artefacts (skipped when absent)


REAL_HY3D_SET = Path(os.environ.get("OC", "/home/cereal/omni-campaign")) / "distill/corpus/media/hy3d"


@pytest.mark.skipif(not (REAL_HY3D_SET / "items.jsonl").is_file(), reason="no distill HY3D set on this host")
def test_real_hy3d_set_imports(tmp_path) -> None:
    out = tmp_path / "hy3d_real.json"
    assert MB.main(["corpus-import", "--set-dir", str(REAL_HY3D_SET), "--kind", "hy3d_images", "--out", str(out)]) == 0
    body = MB.read_corpus(out)
    assert body["n_items"] == 256 and len({i["sha256"] for i in body["items"]}) == 256


S7_CAPTURE_JSON = Path(os.environ.get("OC", "/home/cereal/omni-campaign")) / "composite/s7/pairs/qvl32_vit_L13/parent_capture.json"


@pytest.mark.skipif(not S7_CAPTURE_JSON.is_file(), reason="no s7 QVL capture on this host")
def test_real_s7_capture_images_fire_the_arm_through_smart_resize() -> None:
    Image = pytest.importorskip("PIL.Image")
    body = json.loads(S7_CAPTURE_JSON.read_text())
    sha = {str(i["file"]): str(i["sha256"]) for i in body["images"]}
    assert len(sha) == 32
    for path, digest in sha.items():
        assert MB.sha256_file(Path(path)) == digest
        with Image.open(path) as im:
            w, h = im.size
        rh, rw = MB.smart_resize(h, w)  # qvl-run default --max-pixels 512*512
        assert (rh, rw) == (512, 512)
        assert MB.s7_forward_slice(body["forwards"], sha, digest, rh, rw) is not None


MINI = os.environ.get("MEDIA_B_MINI_CACHES", "")


@pytest.mark.parametrize("cache_dir", [p for p in MINI.split(",") if p])
def test_sealed_mini_cache_verifies(cache_dir) -> None:
    if not Path(cache_dir, "manifest.json").is_file():
        pytest.skip(f"{cache_dir} not sealed")
    assert MB.verify(Path(cache_dir))["verified"]
    manifest = json.loads(Path(cache_dir, "manifest.json").read_text())
    assert manifest["teacher_integrity"]["equal"] and manifest["finite"]
    assert manifest["config_sha256"] == Path(cache_dir).name and set(manifest["shard_tool_versions"]) <= set(manifest["compat"])
    if manifest["teacher"] == "hy3d_trajectories":
        assert all(s["meta"]["faces"] > 0 for s in manifest["shards"])
        assert manifest["checks"]["reference_arms"] >= 1 and manifest["checks"]["determinism_replays"] >= 1
