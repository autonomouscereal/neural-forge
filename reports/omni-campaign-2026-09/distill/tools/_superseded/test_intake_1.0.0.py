"""test_intake.py -- intake.py 1.0.0: drop-zone rules, splits, groups, near-duplicates, media hashes, the contamination
screen (text + media) and its liveness, determinism, the read-only dry run, and (opt-in) the PostgreSQL path end to end.

CPU only (conftest.py hides every card).  Tests that decode audio need ffmpeg on PATH; the PostgreSQL test needs
INTAKE_PG_TEST=1 and CHIMERA_CORPUS_PG_PASSWORD, and works inside a throw-away intake_test_* schema it drops after.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")

HELD_TEXTS = [
    "At dawn, a ceramic wind chime swayed above the courtyard fountain while the gardener counted the copper pots twice.",
    "The night ferry left the harbour late because the captain insisted on checking every lantern on the lower deck.",
    "Seven quiet engineers rebuilt the old observatory telescope using parts salvaged from a shipwrecked weather station.",
]
CLEAN_TEXT = "Our neighbour painted her bicycle bright yellow last spring and now rides it to the bakery every single morning."


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------


def speechlike(seconds: float, seed: int, rate: int = 16000) -> np.ndarray:
    """A deterministic voiced signal: syllable envelopes over a wandering harmonic pitch plus breath noise."""

    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pitch = 110 + 40 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t) + rng.uniform(0, 30)
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    voiced = sum((0.6 / k) * np.sin(k * phase + rng.uniform(0, 6)) for k in range(1, 12))
    syllables = np.zeros(n)
    pos = 0
    while pos < n:
        length = int(rng.uniform(0.12, 0.35) * rate)
        gap = int(rng.uniform(0.03, 0.15) * rate)
        syllables[pos:pos + length] = np.hanning(len(syllables[pos:pos + length]))
        pos += length + gap
    formant = np.sin(2 * np.pi * rng.uniform(300, 900) * t) * 0.2
    noise = rng.normal(0, 0.03, n)
    return (0.3 * syllables * (voiced + formant) + noise * syllables).astype(np.float32)


def write_wav(path: Path, pcm: np.ndarray, rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(pcm, -1, 1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((data * 32767).astype("<i2").tobytes())
    return path


def write_png(path: Path, seed: int, size: int = 96) -> Path:
    from PIL import Image

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    image = Image.fromarray(base).resize((size, size), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def tree_state(root: Path) -> dict[str, tuple[int, int, str]]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns, intake.file_sha256(p))
    return out


def tiny_registry(corpus_root: Path, *, held_audio: np.ndarray | None = None, held_image: Path | None = None,
                  version: int = 1, texts: list[str] | None = None) -> Path:
    texts = HELD_TEXTS if texts is None else texts
    rows = [{"bank": "bank_t", "item": f"t{i}", "rule": "text", "text": t} for i, t in enumerate(texts)]
    rows.append({"bank": "bank_t", "item": "short0", "rule": "text", "text": "violet harbour lantern sings softly"})
    rows.append({"bank": "codes", "item": "c0", "rule": "literal", "text": "ZQX-4417-KAPPA"})
    media = [{"bank": "bank_m", "item": "m0", "sha256": "ab" * 32, "kind": "audio"}]
    if held_image is not None:
        media.append({"bank": "bank_img", "item": "i0", "sha256": intake.file_sha256(held_image), "kind": "image",
                      "phash": intake.image_hashes(held_image)["phash"]})
    fps = []
    if held_audio is not None:
        fps.append(("bank_audio", "a0", intake.audio_fingerprint(held_audio)))
    body = intake.write_registry(corpus_root, version, rows, media, [], fps, [{"name": "tiny"}], "test")
    return Path(body["path"])


def policy(batch_id: str = "speech/ext", meta: dict | None = None, modality_dir: str = "speech") -> dict:
    return intake.batch_policy(batch_id, meta or {}, modality_dir=modality_dir)


def fake_asset(sha_seed: str, batch: str, modality: str = "speech", **extra) -> dict:
    sha = intake.text_sha256(sha_seed)
    base = {"sha256": sha, "ext": "wav", "bytes": 1, "modality": modality, "batch_id": batch, "path": f"/x/{sha_seed}",
            "store_path": None, "derived_from": None, "duration_s": 4.0, "probe": {"ok": True, "has_audio": True},
            "flags": [], "fp": None, "phash": None, "keyframes": None}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------------------------------------------------
# drop-zone rules
# ---------------------------------------------------------------------------------------------------------------------


def test_classify_and_item_files():
    assert intake.classify_name("clip.wav") == "data"
    assert intake.classify_name("clip.wav.part") == "partial"
    assert intake.classify_name("video.mp4.crdownload") == "partial"
    assert intake.classify_name("_meta.json") == "meta"
    assert intake.classify_name(".hidden.wav") == "ignore"
    assert intake.classify_name("_README.txt") == "ignore"
    assert intake.classify_name("clip.wav.sha256") == "sidecar"
    assert intake.is_item_file("clip.flac", "speech")
    assert not intake.is_item_file("clip.txt", "speech")          # transcript sidecar
    assert not intake.is_item_file("1272-128104.trans.txt", "speech")
    assert not intake.is_item_file("metadata.csv", "speech")
    assert not intake.is_item_file("README", "speech")
    assert intake.is_item_file("notes.txt", "text")
    assert intake.is_item_file("rows.jsonl", "text")


def test_batch_for():
    assert intake.batch_for(("speech", "own-2026-10-15", "a.wav"), "2026-10-15") == ("speech", "speech/own-2026-10-15")
    assert intake.batch_for(("speech", "own-2026-10-15", "sub", "a.wav"), "2026-10-15") == ("speech", "speech/own-2026-10-15")
    assert intake.batch_for(("speech", "a.wav"), "2026-10-15") == ("speech", "speech/2026-10-15-loose")
    assert intake.batch_for(("speech", "_received", "x", "a.wav"), "d") is None
    assert intake.batch_for(("elsewhere", "a.wav"), "d") is None
    assert intake.batch_for(("a.wav",), "d") is None


def test_settle_rule():
    now = 1_000_000.0
    ok = dict(size=10, mtime=now - 600, prev_size=10, prev_mtime=now - 600, stable_since=now - 200, now=now, name="a.wav")
    assert intake.is_settled(**ok)
    assert not intake.is_settled(**{**ok, "stable_since": now - 60})          # second scan < 120 s after the first
    assert not intake.is_settled(**{**ok, "mtime": now - 100, "prev_mtime": now - 100})  # mtime younger than 180 s
    assert not intake.is_settled(**{**ok, "size": 11})                        # still growing
    assert not intake.is_settled(**{**ok, "prev_size": None})                 # first sighting
    assert not intake.is_settled(**{**ok, "name": "a.wav.part"})


def test_batch_policy_owner_and_defaults():
    own = policy("speech/own-2026-10-15")
    assert own["owner"] and own["consent"] == "owner" and own["speaker"] == "cyril" and own["group_by"] == "session"
    assert own["session_id"] == "speech/own-2026-10-15"
    meta_owner = policy("speech/kitchen-notes", {"consent": "owner", "session": "s7"})
    assert meta_owner["owner"] and meta_owner["session_id"] == "s7"
    default = policy("images/screens")
    assert default["consent"] == "unspecified" and "STAGE_D_RULINGS 2026-10-14" in default["licence"]
    assert default["group_by"] == "batch"
    licensed = policy("speech/libri", {"consent": "third_party_licensed", "licence": "CC BY 4.0", "group_by": "speaker"})
    assert licensed["consent"] == "third_party_licensed" and licensed["group_by"] == "speaker" and not licensed["flags"]
    odd = policy("speech/x", {"consent": "third_party_licensed", "colour": "blue"})
    assert "licence_missing" in odd["flags"] and any(f.startswith("meta_unknown_keys") for f in odd["flags"])


# ---------------------------------------------------------------------------------------------------------------------
# splits and groups
# ---------------------------------------------------------------------------------------------------------------------


def test_general_split_deterministic_and_proportional():
    ids = [intake.text_sha256(str(i)) for i in range(20000)]
    first = [intake.general_split(g) for g in ids]
    assert first == [intake.general_split(g) for g in ids]
    counts = {s: first.count(s) / len(first) for s in ("train", "val", "test")}
    assert abs(counts["train"] - 0.90) < 0.01 and abs(counts["val"] - 0.05) < 0.01 and abs(counts["test"] - 0.05) < 0.01


def test_owner_session_split_rules():
    assert intake.owner_session_splits(["only"]) == {"only": "test"}       # a lone session is held
    sessions = [f"s{i}" for i in range(12)]
    out = intake.owner_session_splits(sessions)
    assert set(out.values()) == {"train", "test"}
    assert intake.owner_session_splits(list(reversed(sessions))) == out    # order independent
    two = intake.owner_session_splits(["a", "b"])
    assert sorted(two.values()) == ["test", "train"]                       # >= 1 each side
    # sealed sessions never move, even when the forced rule would pick them
    fixed = {"a": "train", "b": "train"}
    out2 = intake.owner_session_splits(["a", "b", "c"], fixed)
    assert out2["a"] == "train" and out2["b"] == "train" and out2["c"] == "test"
    pct = {s: intake.hash_pct(intake.OWNER_SESSION_SALT, s) for s in sessions}
    for s, split in out.items():
        natural = "train" if pct[s] < 80 else "test"
        assert split == natural or split == "test" or split == "train"


def test_groups_link_same_wording_and_inherit_known():
    a = {"identity": ["asset:1", "wording:x", "group:batch:b1"]}
    b = {"identity": ["asset:2", "wording:x", "group:batch:b2"]}
    c = {"identity": ["asset:3", "group:batch:b3"]}
    intake.assign_groups([a, b, c], {})
    assert a["group_id"] == b["group_id"] != c["group_id"]
    d = {"identity": ["asset:4", "wording:x"]}
    intake.assign_groups([d], {"wording:x": ("g" * 64, "test")})
    assert d["group_id"] == "g" * 64 and d["group_split"] == "test"
    e = {"identity": ["asset:5", "wording:y"], "fixed_split": "train"}
    f = {"identity": ["asset:6", "wording:y"], "fixed_split": "test"}
    intake.assign_groups([e, f], {})
    assert e["split_conflict"] and e["group_split"] == "test" and f["group_split"] == "test"


# ---------------------------------------------------------------------------------------------------------------------
# near duplicates and hashes
# ---------------------------------------------------------------------------------------------------------------------


def test_minhash_near_duplicates():
    base = " ".join(f"word{i}" for i in range(200))
    near = base.replace("word50", "changed").replace("word150", "other")
    far = " ".join(f"token{i}" for i in range(200))
    sa, sn, sf = (intake.minhash_signature(t) for t in (base, near, far))
    assert intake.jaccard_estimate(sa, sn) >= 0.5 > intake.jaccard_estimate(sa, sf)
    index = intake.NearDupIndex()
    index.add("base", sa)
    assert index.query(sn)[0] == "base"
    assert index.query(sf) is None
    assert np.array_equal(intake.minhash_signature(base), sa)               # deterministic


def test_fingerprint_chunking_is_exact_and_silence_is_masked(monkeypatch):
    pcm = speechlike(30.0, seed=4)
    whole = intake.audio_fingerprint(pcm)
    monkeypatch.setattr(intake, "FP_CHUNK_FRAMES", 300)
    assert np.array_equal(intake.audio_fingerprint(pcm), whole)            # block boundaries change nothing
    gap = np.concatenate([pcm[:32000], np.zeros(32000, np.float32), pcm[:32000]])
    fp = intake.audio_fingerprint(gap)
    assert (fp == 0).sum() >= 200                                          # ~2 s of digital silence carries no bits
    silence_a = np.concatenate([speechlike(1.0, 50), np.zeros(48000, np.float32)])
    silence_b = np.concatenate([speechlike(1.0, 51), np.zeros(48000, np.float32)])
    index = intake.AudioFpIndex()
    index.add("x", "a", intake.audio_fingerprint(silence_a))
    assert index.match(intake.audio_fingerprint(silence_b)) is None        # shared silent tails never match


def test_audio_fingerprint_matches_variants_and_rejects_others():
    pcm = speechlike(6.0, seed=1)
    fp = intake.audio_fingerprint(pcm)
    index = intake.AudioFpIndex()
    index.add("held", "a0", fp)
    assert index.match(intake.audio_fingerprint(pcm * 0.3))                    # gain
    assert index.match(intake.audio_fingerprint(pcm[4000:]))                   # trimmed start
    assert index.match(intake.audio_fingerprint(np.concatenate([np.zeros(8000, np.float32), pcm])))  # padded
    noisy = pcm + np.random.default_rng(9).normal(0, 0.003, pcm.size).astype(np.float32)
    assert index.match(intake.audio_fingerprint(noisy))
    assert index.match(intake.audio_fingerprint(speechlike(6.0, seed=2))) is None   # different "speaker"
    assert index.match(intake.audio_fingerprint(pcm), exclude_source="held") is None


@needs_ffmpeg
def test_audio_fingerprint_survives_resample_and_mp3(tmp_path):
    pcm = speechlike(5.0, seed=3, rate=24000)
    src = write_wav(tmp_path / "src.wav", pcm, rate=24000)
    mp3 = tmp_path / "re.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ar", "22050", "-b:a", "64k", str(mp3)], check=True)
    index = intake.AudioFpIndex()
    index.add("held", "a0", intake.audio_fingerprint(intake.decode_pcm(src)))
    hit = index.match(intake.audio_fingerprint(intake.decode_pcm(mp3)))
    assert hit and hit["ber"] <= intake.FP_MAX_BER


def test_phash_survives_resize_and_separates_images(tmp_path):
    from PIL import Image

    a = write_png(tmp_path / "a.png", seed=1)
    with Image.open(a) as im:
        im.resize((61, 61)).convert("RGB").save(tmp_path / "a_small.jpg", quality=70)
    b = write_png(tmp_path / "b.png", seed=2)
    ha, hs, hb = (intake.image_hashes(p)["phash"] for p in (a, tmp_path / "a_small.jpg", b))
    assert intake.hamming64(ha, hs) <= intake.PHASH_MAX_HAMMING
    assert intake.hamming64(ha, hb) > intake.PHASH_MAX_HAMMING


def test_plan_segments():
    assert intake.plan_segments(10.0, []) == [(0.0, 10.0)]
    segs = intake.plan_segments(65.0, [(9.0, 10.0), (25.0, 26.0), (40.0, 41.0)])
    assert all(intake.SEG_MIN_S <= e - s <= intake.SEG_MAX_S for s, e in segs)
    assert segs[0] == (0.0, 9.0)
    long_run = intake.plan_segments(50.0, [])
    assert long_run[0] == (0.0, 20.0) and all(e - s <= 20.0 for s, e in long_run)
    assert intake.plan_segments(2.0, []) == []                               # too short to keep alone


def test_transcript_agreement():
    assert intake.word_error_rate("the cat sat", "the cat sat") == 0.0
    assert intake.transcript_verdict("the small wooden boat is moving", "the small wooden boat is moving") == (1.0, "auto")
    assert intake.transcript_verdict("the small wooden boat is moving", "a tall wooden goat was moving")[1] == "review"


def test_sidecar_transcripts(tmp_path):
    libri = tmp_path / "libri" / "84" / "121123"
    write_wav(libri / "84-121123-0000.wav", speechlike(1.0, 1))
    (libri / "84-121123.trans.txt").write_text("84-121123-0000 GO DO YOU HEAR\n", encoding="utf-8")
    lj = tmp_path / "lj"
    write_wav(lj / "wavs" / "LJ001-0001.wav", speechlike(1.0, 2))
    (lj / "metadata.csv").write_text("LJ001-0001|Printing, in 1820|Printing, in eighteen twenty\n", encoding="utf-8")
    own = tmp_path / "own"
    write_wav(own / "memo.wav", speechlike(1.0, 3))
    (own / "memo.txt").write_text("hello there\n", encoding="utf-8")
    found = intake.sidecar_transcripts(tmp_path / "libri", list((tmp_path / "libri").rglob("*.wav")))
    assert found[str(libri / "84-121123-0000.wav")] == {"transcript": "GO DO YOU HEAR", "tool": "LibriSpeech 84-121123.trans.txt", "speaker": "LS-84"}
    found = intake.sidecar_transcripts(lj, list(lj.rglob("*.wav")))
    assert found[str(lj / "wavs" / "LJ001-0001.wav")]["transcript"] == "Printing, in eighteen twenty"
    found = intake.sidecar_transcripts(own, [own / "memo.wav"])
    assert found[str(own / "memo.wav")]["transcript"] == "hello there"


def test_review_merge_keeps_cyrils_lines():
    current = [{"item_id": "b", "transcript": "draft b", "reasons": ["x"], "status": "review", "modality": "speech"},
               {"item_id": "a", "transcript": None, "reasons": ["y"], "status": "review", "modality": "image"}]
    existing = [{"item_id": "b", "accepted": None, "transcript": "his correction"},
                {"item_id": "gone", "accepted": True, "transcript": "applied already"}, "garbage"]
    merged = intake.merge_review_lines(existing, current)
    assert [m["item_id"] for m in merged] == ["a", "b"]
    assert merged[1] == {"item_id": "b", "accepted": None, "transcript": "his correction"}
    assert merged[0]["accepted"] is None and merged[0]["modality"] == "image"


def test_text_extraction(tmp_path):
    jl = tmp_path / "rows.jsonl"
    jl.write_text(json.dumps({"instruction": "Say hi", "response": "Hi!"}) + "\n" + json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}) + "\n", encoding="utf-8")
    rows, problem = intake.extract_text_rows(jl, "jsonl")
    assert problem is None and rows[0] == {"instruction": "Say hi", "response": "Hi!"} and len(rows) == 2
    big = tmp_path / "big.txt"
    big.write_text("\n\n".join(" ".join(["w"] * 400) for _ in range(10)), encoding="utf-8")
    rows, _ = intake.extract_text_rows(big, "txt")
    assert len(rows) == 4 and all(len(r["response"].split()) <= intake.TEXT_CHUNK_WORDS for r in rows)  # 3+3+3+1 paragraphs
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    assert intake.extract_text_rows(bad, "jsonl") == ([], "bad_jsonl")


# ---------------------------------------------------------------------------------------------------------------------
# registry, screen, liveness, determinism
# ---------------------------------------------------------------------------------------------------------------------


def test_registry_seal_and_tamper(tmp_path):
    path = tiny_registry(tmp_path, held_audio=speechlike(4.0, 5))
    holdouts = intake.Holdouts.load(path)
    assert holdouts.version == 1 and holdouts.sha256
    with pytest.raises(intake.Refusal):
        tiny_registry(tmp_path)                                            # append-only
    texts = path.parent / "registry_v1" / "texts.jsonl"
    texts.write_text(texts.read_text(encoding="utf-8") + json.dumps({"bank": "x", "item": "y", "rule": "text", "text": "z"}) + "\n", encoding="utf-8")
    with pytest.raises(intake.Refusal):
        intake.Holdouts.load(path)


def test_liveness_live_and_dead(tmp_path):
    live_reg = tiny_registry(tmp_path / "live", held_audio=speechlike(4.0, 5), held_image=write_png(tmp_path / "h.png", 4))
    holdouts, report = intake.load_holdouts_live(tmp_path / "live", str(live_reg))
    assert report["live"] and report["media"]["totals"]["audio_caught"] == 1 and report["media"]["totals"]["phash_caught"] == 1
    dead_reg = tiny_registry(tmp_path / "dead", texts=["too short to probe"])
    with pytest.raises(intake.Refusal):
        intake.load_holdouts_live(tmp_path / "dead", str(dead_reg))        # no long item -> the screen cannot prove itself


def test_contamination_screen_in_planner(tmp_path):
    held_pcm = speechlike(5.0, seed=11)
    held_png = write_png(tmp_path / "held.png", 7)
    reg = tiny_registry(tmp_path, held_audio=held_pcm, held_image=held_png)
    holdouts = intake.Holdouts.load(reg)
    known = intake.Known()
    batch = "speech/ext"
    pol = {batch: policy(batch), "images/x": policy("images/x", modality_dir="images"), "text/t": policy("text/t", modality_dir="text")}
    from PIL import Image
    with Image.open(held_png) as im:
        im.resize((80, 80)).convert("RGB").save(tmp_path / "held_copy.jpg", quality=80)
    assets = [
        fake_asset("verbatim", batch),
        fake_asset("rewrapped", batch),
        fake_asset("mediasha", batch),
        fake_asset("audiofp", batch, fp=intake.audio_fingerprint(held_pcm * 0.5)),
        fake_asset("clean", batch, fp=intake.audio_fingerprint(speechlike(5.0, seed=12))),
        fake_asset("short", batch),
        fake_asset("code", batch),
        fake_asset("img", "images/x", modality="image", phash=intake.image_hashes(tmp_path / "held_copy.jpg")["phash"]),
        fake_asset("doc", "text/t", modality="text"),
    ]
    assets[2]["sha256"] = "ab" * 32
    transcripts = {
        assets[0]["sha256"]: {"transcript": HELD_TEXTS[0], "tool": "t"},
        assets[1]["sha256"]: {"transcript": "Exported:\n" + "\n".join(HELD_TEXTS[1].upper().split()), "tool": "t"},
        assets[4]["sha256"]: {"transcript": CLEAN_TEXT, "tool": "t"},
        assets[5]["sha256"]: {"transcript": "then the violet harbour lantern sings softly again", "tool": "t"},
        assets[6]["sha256"]: {"transcript": "the code is zqx-4417-kappa today", "tool": "t"},
    }
    text_rows = {assets[8]["sha256"]: [{"instruction": "Summarise", "response": HELD_TEXTS[2] + " And more words follow here."}]}
    items = intake.plan_items(assets, pol, transcripts, text_rows, holdouts, known)
    by_asset = {it["asset_sha256"]: it for it in items}
    rules = {name: {h["rule"] for h in by_asset[a["sha256"]]["screen"]["text"] + by_asset[a["sha256"]]["screen"]["media"]}
             for name, a in zip(("verbatim", "rewrapped", "mediasha", "audiofp", "clean", "short", "code", "img", "doc"), assets)}
    statuses = {name: by_asset[a["sha256"]]["status"] for name, a in zip(("verbatim", "rewrapped", "mediasha", "audiofp", "clean", "short", "code", "img", "doc"), assets)}
    assert statuses["verbatim"] == "contaminated" and "ngram13" in rules["verbatim"]
    assert statuses["rewrapped"] == "contaminated"
    assert statuses["mediasha"] == "contaminated" and "media_sha256" in rules["mediasha"]
    assert statuses["audiofp"] == "contaminated" and "audio_fingerprint" in rules["audiofp"]
    assert statuses["short"] == "contaminated" and "short_item" in rules["short"]
    assert statuses["code"] == "contaminated" and "literal" in rules["code"]
    assert statuses["img"] == "contaminated" and "phash" in rules["img"]
    assert statuses["doc"] == "contaminated"
    assert statuses["clean"] in ("admitted", "held") and not rules["clean"]


def test_planner_dedupe_and_owner_sessions(tmp_path):
    holdouts = intake.Holdouts.load(tiny_registry(tmp_path))
    known = intake.Known()
    pcm = speechlike(5.0, seed=21)
    pol = {b: policy(b) for b in ("speech/own-a", "speech/own-b", "speech/own-c", "speech/ext")}
    pol["text/t"] = policy("text/t", modality_dir="text")
    assets = [
        fake_asset("orig", "speech/own-a", fp=intake.audio_fingerprint(pcm)),
        fake_asset("resampled-copy", "speech/own-b", fp=intake.audio_fingerprint(pcm * 0.8)),   # same audio, other bytes
        fake_asset("c1", "speech/own-c", fp=intake.audio_fingerprint(speechlike(5.0, 22))),
        fake_asset("t1", "text/t", modality="text"),
        fake_asset("t2", "text/t", modality="text"),
    ]
    long_text = " ".join(f"alpha{i}" for i in range(120))
    rows = {assets[3]["sha256"]: [{"instruction": "", "response": long_text}],
            assets[4]["sha256"]: [{"instruction": "", "response": long_text.replace("alpha60", "beta")}]}
    trans = {a["sha256"]: {"transcript": f"owner sentence number {i} about the weather today", "tool": "t"} for i, a in enumerate(assets[:3])}
    items = intake.plan_items(assets, pol, trans, rows, holdouts, known)
    st = {it["asset_sha256"]: it for it in items}
    statuses = sorted(st[a["sha256"]]["status"] for a in assets[:2])
    assert "duplicate_near" in statuses                                     # the second copy of one recording
    text_statuses = sorted(st[a["sha256"]]["status"] for a in assets[3:])
    assert "duplicate_near" in text_statuses
    own = [it for it in items if it["owner"]]
    assert {it["voice"] for it in own} == {"owner-real"}
    splits = {it["session_id"]: it["split"] for it in own}
    assert set(splits.values()) <= {"train", "test"} and "test" in splits.values() and "train" in splits.values()


def test_plan_is_deterministic_under_input_order(tmp_path):
    holdouts = intake.Holdouts.load(tiny_registry(tmp_path))
    pol = {f"speech/b{i}": policy(f"speech/b{i}", {"group_by": "file"}) for i in range(5)}
    assets = [fake_asset(f"a{i}", f"speech/b{i % 5}", fp=intake.audio_fingerprint(speechlike(3.0, 100 + i))) for i in range(30)]
    trans = {a["sha256"]: {"transcript": f"sentence {i % 7} spoken with care by the reader", "tool": "t"} for i, a in enumerate(assets)}

    def snapshot(order):
        items = intake.plan_items([assets[i] for i in order], pol, trans, {}, holdouts, intake.Known())
        return sorted((it["item_id"], it["status"], it["split"], it["group_id"], it.get("duplicate_of")) for it in items)

    first = snapshot(range(30))
    order = list(range(30))
    random.Random(5).shuffle(order)
    assert snapshot(order) == first
    env = dict(os.environ, PYTHONHASHSEED="123")
    code = ("import sys,json;sys.path.insert(0,%r);import intake;"
            "print(json.dumps([intake.general_split(intake.text_sha256(str(i))) for i in range(50)]))") % str(Path(intake.__file__).parent)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [intake.general_split(intake.text_sha256(str(i))) for i in range(50)]


# ---------------------------------------------------------------------------------------------------------------------
# dry run: read-only over a drop zone
# ---------------------------------------------------------------------------------------------------------------------


@needs_ffmpeg
def test_dry_run_is_read_only_and_reports(tmp_path):
    drop = tmp_path / "drop"
    held_pcm = speechlike(4.0, seed=31)
    write_wav(drop / "speech" / "own-2026-10-15" / "memo1.wav", speechlike(4.0, 32))
    (drop / "speech" / "own-2026-10-15" / "memo1.txt").write_text("this is my first memo about the garden shed", encoding="utf-8")
    write_wav(drop / "speech" / "own-2026-10-16" / "memo2.wav", speechlike(4.0, 33))
    write_wav(drop / "speech" / "own-2026-10-16" / "held_copy.wav", held_pcm * 0.7)
    write_wav(drop / "speech" / "loose.wav", speechlike(3.0, 34))
    (drop / "text" / "notes").mkdir(parents=True)
    (drop / "text" / "notes" / "rows.jsonl").write_text(json.dumps({"instruction": "q", "response": HELD_TEXTS[0]}) + "\n"
                                                        + json.dumps({"instruction": "q2", "response": CLEAN_TEXT}) + "\n", encoding="utf-8")
    write_png(drop / "images" / "x.png", 3)
    (drop / "speech" / "partial.wav.part").write_bytes(b"xx")
    corpus = tmp_path / "corpus"
    tiny_registry(corpus, held_audio=held_pcm)
    before = tree_state(tmp_path)
    out = tmp_path / "out"
    args = intake.parse_args(["run", "--dry-run", "--drop-root", str(drop), "--corpus-root", str(corpus), "--out", str(out),
                              "--queues", str(tmp_path / "queues"), "--workers", "1"])
    env_pw = os.environ.pop(intake.PG_PASSWORD_ENV, None)
    try:
        report = intake.cmd_run(args)
    finally:
        if env_pw is not None:
            os.environ[intake.PG_PASSWORD_ENV] = env_pw
    after = tree_state(tmp_path)
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed and all(k.startswith("out" + os.sep) or k.startswith("out/") for k in changed), changed
    summary = report["summary"]
    assert summary["assets"] == 6
    statuses = {Path(it["asset_path"]).name: it["status"] for it in map(json.loads, (out / "dry_run_items.jsonl").read_text(encoding="utf-8").splitlines())
                if it.get("row") in (None, 0)}
    assert statuses["held_copy.wav"] == "contaminated"
    assert statuses["memo2.wav"] == "candidate"                            # no transcript yet
    assert statuses["rows.jsonl"] == "contaminated"
    assert report["policies"]["speech/own-2026-10-15"]["consent"] == "owner"
    assert intake.Holdouts.load(Path(report["registry"]["path"])).sha256 == report["registry"]["sha256"]
    sealed = json.loads((out / "dry_run_report.json").read_text(encoding="utf-8"))
    assert sealed["report_sha256"] == intake.canonical_sha256({k: v for k, v in sealed.items() if k != "report_sha256"})


# ---------------------------------------------------------------------------------------------------------------------
# PostgreSQL end to end (opt-in): run twice (settle), idempotent third run, originals untouched, seal, verify, resolve
# ---------------------------------------------------------------------------------------------------------------------

PG_READY = os.environ.get("INTAKE_PG_TEST") == "1" and bool(os.environ.get(intake.PG_PASSWORD_ENV))


@pytest.mark.skipif(not (PG_READY and HAVE_FFMPEG), reason="INTAKE_PG_TEST=1, CHIMERA_CORPUS_PG_PASSWORD and ffmpeg needed")
def test_postgres_end_to_end(tmp_path, monkeypatch):
    if not hasattr(os, "fork"):
        pytest.skip("server-side test (fcntl locks)")
    schema = f"intake_test_{uuid.uuid4().hex[:12]}"
    import psycopg2

    with intake.pg_session() as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")  # name is generated above, never user data
        conn.commit()
    monkeypatch.setenv(intake.PG_SCHEMA_ENV, schema)
    try:
        drop, corpus, queues, archive = tmp_path / "drop", tmp_path / "corpus", tmp_path / "queues", tmp_path / "archive"
        held_pcm = speechlike(4.0, seed=41)
        write_wav(drop / "speech" / "own-s1" / "a.wav", speechlike(4.0, 42))
        (drop / "speech" / "own-s1" / "a.txt").write_text("my first recorded sentence about the river walk", encoding="utf-8")
        write_wav(drop / "speech" / "own-s2" / "b.wav", speechlike(4.0, 43))
        (drop / "speech" / "own-s2" / "b.txt").write_text("a second sentence recorded on another evening at home", encoding="utf-8")
        write_wav(drop / "speech" / "own-s2" / "leak.wav", held_pcm)
        write_wav(drop / "speech" / "own-s1" / "c.mp3", speechlike(3.0, 44))          # WAV bytes behind an .mp3 name -> review
        (drop / "text" / "t1").mkdir(parents=True)
        (drop / "text" / "t1" / "rows.jsonl").write_text(json.dumps({"instruction": "q", "response": CLEAN_TEXT}) + "\n", encoding="utf-8")
        old = time.time() - 3600
        for p in drop.rglob("*"):
            if p.is_file():
                os.utime(p, (old, old))
        reg = tiny_registry(corpus, held_audio=held_pcm)
        assert intake.main(["dropzone-init", "--drop-root", str(drop)]) == 0
        before = tree_state(drop)
        base = ["--drop-root", str(drop), "--corpus-root", str(corpus), "--queues", str(queues), "--archive-root", str(archive)]
        assert intake.main(["init-db", *base]) == 0
        with intake.pg_session() as conn:
            intake.PgStore(conn).register_holdouts(json.loads(reg.read_text(encoding="utf-8")), str(reg), "test")
            with conn.cursor() as cur:
                cur.execute("INSERT INTO corpus_version (version_no, status, tool_version, tool_sha256) VALUES (1, 'sealed', 'adopted', %s)", ("0" * 64,))
            conn.commit()
        now = time.time()
        assert intake.main(["run", *base, "--now", str(now)]) == 0                 # first sighting: nothing settles
        with intake.pg_session() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM item")
            assert cur.fetchone()[0] == 0
        assert intake.main(["run", *base, "--now", str(now + 200)]) == 0           # second scan >= 120 s later
        with intake.pg_session() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*), count(*) FILTER (WHERE status = 'contaminated'), count(*) FILTER (WHERE status = 'review') FROM item")
            total, contaminated, review = cur.fetchone()
            cur.execute("SELECT count(*) FROM asset WHERE derived_from IS NOT NULL")
            derived = cur.fetchone()[0]
        assert (total, contaminated, review, derived) == (5, 1, 1, 8)                # 4 audio x (16k mono + 32k stereo)
        review_file = drop / "_review" / "speech" / "own-s1" / "transcripts.jsonl"
        lines = [json.loads(x) for x in review_file.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 1 and lines[0]["accepted"] is None and any(r.startswith("extension_mismatch") for r in lines[0]["reasons"])
        lines[0]["transcript"] = "an edited line that must survive the next pass"
        review_file.write_text(json.dumps(lines[0]) + "\n", encoding="utf-8")
        assert intake.main(["run", *base, "--now", str(now + 400)]) == 0           # idempotent; the undecided edit stays
        assert json.loads(review_file.read_text(encoding="utf-8"))["transcript"] == "an edited line that must survive the next pass"
        with intake.pg_session() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM item")
            assert cur.fetchone()[0] == 5
        lines[0]["accepted"] = True
        review_file.write_text(json.dumps(lines[0]) + "\n", encoding="utf-8")
        assert intake.main(["run", *base, "--now", str(now + 600)]) == 0           # decision applied
        assert review_file.read_text(encoding="utf-8") == ""
        assert tree_state(drop) == {**before, **{k: v for k, v in tree_state(drop).items() if k.startswith("_")}}  # originals untouched
        assert len([p for p in (drop / "_received").rglob("*.*") if p.name != "_README.txt"]) == 5
        with intake.pg_session() as conn, conn.cursor() as cur:
            cur.execute("SELECT i.status, i.split, i.transcript, a.batch_id FROM item i JOIN asset a ON a.sha256 = i.asset_sha256 ORDER BY 4, 3")
            states = cur.fetchall()
        own1 = [s for s in states if s[3] == "speech/own-s1"]
        own2 = [s for s in states if s[3] == "speech/own-s2" and s[0] != "contaminated"]
        assert len(own1) == 2 and {s[0] for s in own1} <= {"admitted", "held"}, states
        assert "an edited line that must survive the next pass" in {s[2] for s in own1}, states
        assert {own1[0][1], own2[0][1]} == {"train", "test"}, states             # two sessions: one each side
        assert [s[0] for s in states if s[3] == "text/t1"][0] in ("admitted", "held"), states
        assert intake.main(["seal", *base, "--version", "2"]) == 0
        assert intake.main(["seal", *base, "--version", "3"]) == 2                 # nothing new: refused
        assert intake.main(["verify", *base, "--version", "2"]) == 0
        out = tmp_path / "resolved.jsonl"
        assert intake.main(["resolve", *base, "--version", "2", "--split", "train", "--out", str(out)]) == 0
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert all(r["split"] == "train" for r in rows)
        manifest = json.loads((corpus / "v2" / "manifest.json").read_text(encoding="utf-8"))
        victim = Path(next(iter(manifest["store_files"].values()))["path"])
        victim.write_bytes(victim.read_bytes() + b"x")
        assert intake.main(["verify", *base, "--version", "2"]) == 2
    finally:
        monkeypatch.delenv(intake.PG_SCHEMA_ENV, raising=False)
        with intake.pg_session() as conn, conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.commit()
        del psycopg2


def test_dropzone_init_idempotent_and_readmes_never_items(tmp_path):
    root = tmp_path / "zone"
    first = intake.cmd_dropzone_init(intake.parse_args(["dropzone-init", "--drop-root", str(root)]))
    assert len(first["written"]) == 1 + len(intake.README_FOLDER)
    assert intake.cmd_dropzone_init(intake.parse_args(["dropzone-init", "--drop-root", str(root)]))["written"] == []
    text = (root / "_README.txt").read_text(encoding="utf-8")
    assert "NEVER MODIFIED, MOVED, RENAMED OR DELETED" in text
    for name in intake.README_FOLDER:
        assert "never modified, moved, renamed or deleted" in (root / name / "_README.txt").read_text(encoding="utf-8")
    assert [p for p, _ in intake.walk_drop_root(root) if intake.is_item_file(p.name, p.relative_to(root).parts[0])] == []


def test_run_refuses_an_uninitialised_or_unmounted_drop_zone(tmp_path):
    args = intake.parse_args(["run", "--drop-root", str(tmp_path / "not-mounted"), "--corpus-root", str(tmp_path)])
    with pytest.raises(intake.Refusal):
        intake.cmd_run(args)
