"""test_intake.py -- intake.py 1.1.1: drop-zone rules, splits, groups, near-duplicates, media hashes, the contamination
screen (text + media) and its liveness, determinism, the read-only dry run, and (opt-in) the PostgreSQL path end to end.
1.1.0 adds: owner sessions split by session hash alone (multi-pass stable, never split), loose speech = owner memos,
long-recording transcript_ref, held near-duplicate refusal, registry v2 banks (tool-call bank, Codex recovery val/test,
BFCL-62 + twins, held corpus rows), malformed sidecars / pdftotext timeout, the seal state machine, and on PostgreSQL:
crash mid-pass then recovery, a contained Refusal, chunked passes (file cap, ingest budget, derivative budget),
intake_run ok/errors, failed-seal recovery and a sealed-only rebuild-index.

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
    sessions = [f"s{i}" for i in range(200)]
    out = intake.owner_session_splits(sessions)
    for s, split in out.items():                                            # the session hash ALONE decides
        assert split == ("train" if intake.hash_pct(intake.OWNER_SESSION_SALT, s) < 80 else "test")
        assert intake.owner_session_splits([s]) == {s: split}               # alone, or with any company: the same
        assert intake.owner_session_splits([s, "x", "y"])[s] == split
    assert intake.owner_session_splits(list(reversed(sessions))) == out    # order independent
    assert abs(sum(v == "train" for v in out.values()) / len(out) - 0.80) < 0.08
    # 1.0.0 forced moves made these depend on company; 1.1.0: a lone all-train set stays train (reported at seal)
    lone_train = next(s for s in sessions if out[s] == "train")
    assert intake.owner_session_splits([lone_train]) == {lone_train: "train"}
    # a sealed session never moves (a sealed 1.0.0 forced move is kept)
    assert intake.owner_session_splits([lone_train], {lone_train: "test"}) == {lone_train: "test"}


def test_groups_link_same_wording_and_inherit_known():
    a = {"identity": ["asset:1", "wording:x", "group:batch:b1"]}
    b = {"identity": ["asset:2", "wording:x", "group:batch:b2"]}
    c = {"identity": ["asset:3", "group:batch:b3"]}
    intake.assign_groups([a, b, c], {})
    assert a["group_id"] == b["group_id"] != c["group_id"]
    d = {"identity": ["asset:4", "wording:x"]}
    intake.assign_groups([d], {"wording:x": ("g" * 64, "test")})
    assert d["group_id"] == "g" * 64 and d["group_split"] == "test"
    # 1.1.0: an item with its own fixed split (an owner session) keeps it -- a session is never split -- and the one
    # on the LESS held side of a held link goes to review; a free item in the group takes the most-held split
    e = {"identity": ["asset:5", "wording:y"], "fixed_split": "train"}
    f = {"identity": ["asset:6", "wording:y"], "fixed_split": "test"}
    g = {"identity": ["asset:7", "wording:y"]}
    intake.assign_groups([e, f, g], {})
    assert e["group_split"] == "train" and e["split_conflict"]
    assert f["group_split"] == "test" and not f["split_conflict"]
    assert g["group_split"] == "test" and not g["split_conflict"]


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
    for it in own:                                                          # session hash alone
        assert it["split"] == intake.owner_session_split(it["session_id"])


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



def pg_rows(sql: str, params: tuple = ()) -> list[tuple]:
    with intake.pg_session() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@pytest.mark.skipif(not (PG_READY and HAVE_FFMPEG), reason="INTAKE_PG_TEST=1, CHIMERA_CORPUS_PG_PASSWORD and ffmpeg needed")
def test_postgres_crash_recovery_chunking_errors_and_seal_states(tmp_path, monkeypatch):
    """1.1.0 on PostgreSQL: a crash mid-pass (after asset rows, before items) and an interrupt mid-commit are both
    finished by the next pass's recovery sweep; a Refusal on one file is contained; --max-files chunks a drop across
    passes; intake_run records ok / errors / error_text and closes an abandoned row; owner sessions keep their hash
    split across passes; a partial v<N> is moved aside, a seal that died after its rename is completed, and
    rebuild-index skips an unsealed folder."""

    if not hasattr(os, "fork"):
        pytest.skip("server-side test (fcntl locks)")
    schema = f"intake_test_{uuid.uuid4().hex[:12]}"
    with intake.pg_session() as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")  # name is generated above, never user data
        conn.commit()
    monkeypatch.setenv(intake.PG_SCHEMA_ENV, schema)
    try:
        drop, corpus, queues, archive = tmp_path / "drop", tmp_path / "corpus", tmp_path / "queues", tmp_path / "archive"
        write_wav(drop / "speech" / "Recording 2026-10-12 0812.wav", speechlike(4.0, 61))             # loose owner memo
        write_wav(drop / "speech" / "own-s1" / "a.wav", speechlike(4.0, 62))
        (drop / "speech" / "own-s1" / "a.txt").write_text("the first owner sentence about the river path today", encoding="utf-8")
        (drop / "speech" / "own-s1" / "_transcripts.jsonl").write_bytes(b'{"file": "zz.wav", "transcript": "x \xff"}\n{broken\n')
        write_wav(drop / "speech" / "own-s2" / "b.wav", speechlike(4.0, 63))
        (drop / "speech" / "own-s2" / "b.txt").write_text("a second owner sentence recorded on another quiet evening", encoding="utf-8")
        write_wav(drop / "speech" / "own-s2" / "b2.wav", speechlike(4.0, 64))
        (drop / "speech" / "own-s2" / "b2.txt").write_text("a third owner sentence in the same evening session at home", encoding="utf-8")
        (drop / "text" / "t1").mkdir(parents=True)
        (drop / "text" / "t1" / "rows.jsonl").write_text(json.dumps({"instruction": "q", "response": CLEAN_TEXT}) + "\n", encoding="utf-8")
        (drop / "text" / "t2").mkdir(parents=True)
        (drop / "text" / "t2" / "bad.txt").write_text("this file's copy is refused by the test harness\n", encoding="utf-8")
        old = time.time() - 3600
        for p in drop.rglob("*"):
            if p.is_file():
                os.utime(p, (old, old))
        reg = tiny_registry(corpus)
        assert intake.main(["dropzone-init", "--drop-root", str(drop)]) == 0
        base = ["--drop-root", str(drop), "--corpus-root", str(corpus), "--queues", str(queues), "--archive-root", str(archive)]
        assert intake.main(["init-db", *base]) == 0
        with intake.pg_session() as conn:
            intake.PgStore(conn).register_holdouts(json.loads(reg.read_text(encoding="utf-8")), str(reg), "test")
            with conn.cursor() as cur:
                cur.execute("INSERT INTO corpus_version (version_no, status, tool_version, tool_sha256) VALUES (1, 'sealed', 'adopted', %s)", ("0" * 64,))
            conn.commit()
        real_copy = intake.copy_verified

        def refusing_copy(source, destination, sha):
            if Path(source).name == "bad.txt":
                raise intake.Refusal(f"{source} changed while it was being copied")       # what a racing writer causes
            return real_copy(source, destination, sha)

        monkeypatch.setattr(intake, "copy_verified", refusing_copy)
        now = time.time()
        assert intake.main(["run", *base, "--now", str(now)]) == 0                               # pass 1: first sighting
        # pass 1b: settled, but no budget left: every file waits (still settled) for the next pass
        assert intake.main(["run", *base, "--now", str(now + 150), "--budget-s", "0"]) == 0
        assert json.loads((queues / "intake.status").read_text(encoding="utf-8"))["counts"]["deferred_files"] == 6
        assert pg_rows("SELECT count(*) FROM asset")[0][0] == 0
        # pass 2: crash after two asset rows, before any item (plan_items raises) ----------------------------------
        real_plan = intake.plan_items

        def crashing_plan(*args, **kwargs):
            raise RuntimeError("simulated crash between asset rows and items")

        monkeypatch.setattr(intake, "plan_items", crashing_plan)
        with pytest.raises(RuntimeError):
            intake.main(["run", *base, "--now", str(now + 200), "--max-files", "2"])
        monkeypatch.setattr(intake, "plan_items", real_plan)
        assert pg_rows("SELECT count(*) FROM asset WHERE derived_from IS NULL")[0][0] == 2
        assert pg_rows("SELECT count(*) FROM item")[0][0] == 0
        assert pg_rows("SELECT count(*) FROM intake_file WHERE status = 'hashed'")[0][0] == 2
        ok, err = pg_rows("SELECT ok, error_text FROM intake_run ORDER BY run_id DESC LIMIT 1")[0]
        assert ok is False and "simulated crash" in err
        assert json.loads((queues / "intake.status").read_text(encoding="utf-8"))["ok"] is False
        # pass 3: recovery sweep finishes both + two new files; an interrupt kills the second commit ------------------
        real_finish = intake.PgStore.finish_asset
        calls = {"n": 0}

        def interrupted_finish(self, subject, items, registry_version):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt("simulated kill during a commit")
            return real_finish(self, subject, items, registry_version)

        monkeypatch.setattr(intake.PgStore, "finish_asset", interrupted_finish)
        with pytest.raises(KeyboardInterrupt):
            intake.main(["run", *base, "--now", str(now + 400), "--max-files", "2"])
        monkeypatch.setattr(intake.PgStore, "finish_asset", real_finish)
        assert pg_rows("SELECT count(*) FROM intake_step WHERE step = 'items' AND status = 'done'")[0][0] == 1
        assert pg_rows("SELECT count(*) FROM asset WHERE derived_from IS NULL")[0][0] == 4
        assert pg_rows("SELECT ok FROM intake_run ORDER BY run_id DESC LIMIT 1")[0][0] is False
        # pass 4: recovers the 3 unfinished assets, ingests the next 2 (bad.txt among the remaining is refused) ------
        assert intake.main(["run", *base, "--now", str(now + 600), "--max-files", "2"]) == 0
        # pass 5: the rest; pass 6: nothing new, idempotent; an abandoned run row is closed --------------------------
        assert intake.main(["run", *base, "--now", str(now + 800)]) == 0
        with intake.pg_session() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO intake_run (mode, tool_sha256) VALUES ('run', %s) RETURNING run_id", ("0" * 64,))
            abandoned = cur.fetchone()[0]
            conn.commit()
        items_before = pg_rows("SELECT count(*) FROM item")[0][0]
        assert intake.main(["run", *base, "--now", str(now + 1000)]) == 0
        assert pg_rows("SELECT count(*) FROM item")[0][0] == items_before == 5
        assert pg_rows("SELECT ok, error_text FROM intake_run WHERE run_id = %s", (abandoned,))[0][0] is False
        assert "abandoned" in pg_rows("SELECT error_text FROM intake_run WHERE run_id = %s", (abandoned,))[0][0]
        with intake.pg_session() as conn:
            assert intake.PgStore(conn).pending_assets() == []
        runs = pg_rows("SELECT ok, errors, error_text FROM intake_run WHERE run_id <> %s ORDER BY run_id", (abandoned,))
        assert [r[0] for r in runs] == [True, True, False, False, True, True, True], runs
        refused = [r for r in runs if r[1]]
        assert len(refused) == 1 and refused[0][1] == 1 and "bad.txt" in refused[0][2]     # contained, the pass went on
        files = dict(pg_rows("SELECT drop_path, status FROM intake_file"))
        assert files[str(drop / "text" / "t2" / "bad.txt")] == "error"
        assert {s for p, s in files.items() if not p.endswith("bad.txt")} == {"archived"}
        assert pg_rows("SELECT count(*) FROM intake_step WHERE step = 'items' AND status = 'done'")[0][0] == 5
        # owner sessions: hash split, one side each, loose memo = owner with its file-date session -------------------
        states = pg_rows("SELECT i.session_id, i.split, i.status, i.consent FROM item i JOIN asset a ON a.sha256 = i.asset_sha256 "
                         "WHERE a.batch_id LIKE 'speech/%%'")
        sides: dict[str, set] = {}
        for session, split, status, consent in states:
            sides.setdefault(session, set()).add(split)
            assert consent == "owner"
        assert sides == {s: {intake.owner_session_split(s)} for s in ("speech/own-s1", "speech/own-s2", "speech/loose-2026-10-12")}
        assert sides["speech/own-s1"] == {"train"} and sides["speech/own-s2"] == {"test"}
        assert pg_rows("SELECT count(*) FROM item WHERE transcript IS NOT NULL AND modality = 'speech'")[0][0] == 3
        # seal: a stale partial v2 (a 1.0.0 seal that died) is moved aside, never blocking ----------------------------
        (corpus / "v2" / "rows").mkdir(parents=True)
        (corpus / "v2" / "rows" / "speech.jsonl").write_text("half written\n", encoding="utf-8")
        assert intake.main(["seal", *base, "--version", "2"]) == 0
        assert [p.name.split("-")[0] for p in (corpus / "_failed_seals").iterdir()] == ["v2"]
        assert intake.main(["verify", *base, "--version", "2"]) == 0
        manifest = json.loads((corpus / "v2" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["owner_sessions"]["by_split"] == {"test": ["speech/own-s2"], "train": ["speech/own-s1"]}
        assert manifest["warnings"] == []
        # the held rows of v2 join the next registry
        entry, texts, *_ = intake.collect_bank({"name": "corpus_heldout", "kind": "corpus_versions_heldout", "corpus_root": str(corpus)})
        v2_rows = [json.loads(line) for f in (corpus / "v2" / "rows").glob("*.jsonl") for line in f.read_text(encoding="utf-8").splitlines()]
        assert entry["corpus_versions"][0]["version"] == 2
        assert entry["corpus_versions"][0]["held_rows"] == sum(1 for r in v2_rows if r["split"] != "train") >= 2
        assert {"a second owner sentence recorded on another quiet evening",
                "a third owner sentence in the same evening session at home"} <= {t["text"] for t in texts}
        assert "the first owner sentence about the river path today" not in {t["text"] for t in texts}   # train stays out
        # seal v3 dies after its rename (before PostgreSQL): the retry completes it instead of refusing forever ------
        real_register = intake.register_sealed_version
        state = {"first": True}

        def dying_register(*args, **kwargs):
            if state["first"]:
                state["first"] = False
                raise RuntimeError("simulated crash after the rename")
            return real_register(*args, **kwargs)

        monkeypatch.setattr(intake, "register_sealed_version", dying_register)
        with pytest.raises(RuntimeError):
            intake.main(["seal", *base, "--version", "3", "--allow-empty"])
        assert (corpus / "v3" / "sealed").is_file() and not pg_rows("SELECT 1 FROM corpus_version WHERE version_no = 3")
        assert intake.main(["seal", *base, "--version", "3", "--allow-empty"]) == 0
        assert pg_rows("SELECT status FROM corpus_version WHERE version_no = 3") == [("sealed",)]
        monkeypatch.setattr(intake, "register_sealed_version", real_register)
        # rebuild-index registers sealed folders only ---------------------------------------------------------------
        shutil.copytree(corpus / "v2", corpus / "v4")
        (corpus / "v4" / "sealed").unlink()
        result = intake.cmd_rebuild_index(intake.parse_args(["rebuild-index", *base]))
        assert result["skipped_unsealed"] == [str(corpus / "v4")]
        assert not pg_rows("SELECT 1 FROM corpus_version WHERE version_no = 4")
        assert intake.main(["resolve", *base, "--version", "3", "--out", str(tmp_path / "r.jsonl")]) == 0
        assert intake.main(["resolve", *base, "--version", "4", "--out", str(tmp_path / "r4.jsonl")]) == 2   # never read
        # a pass whose budget runs out during derivatives defers the rest; the next pass's sweep finishes them -------
        for name, seed in (("x.wav", 71), ("y.wav", 72)):
            write_wav(drop / "speech" / "own-s3" / name, speechlike(4.0, seed))
            os.utime(drop / "speech" / "own-s3" / name, (old, old))
        assert intake.main(["run", *base, "--now", str(now + 1200)]) == 0
        offset = {"s": 0.0}
        monkeypatch.setattr(intake, "pass_clock", lambda: time.time() + offset["s"])
        real_derive = intake.make_derivatives

        def slow_derive(asset, store):
            out = real_derive(asset, store)
            offset["s"] = 10_000.0                                                          # the budget is gone
            return out

        monkeypatch.setattr(intake, "make_derivatives", slow_derive)
        assert intake.main(["run", *base, "--now", str(now + 1400), "--budget-s", "1000"]) == 0
        counts = json.loads((queues / "intake.status").read_text(encoding="utf-8"))["counts"]
        assert counts["ingested"] == 2 and counts["assets_finished"] == 1 and counts["deferred_assets"] == 1, counts
        with intake.pg_session() as conn:
            assert len(intake.PgStore(conn).pending_assets()) == 1
        monkeypatch.setattr(intake, "make_derivatives", real_derive)
        offset["s"] = 0.0
        assert intake.main(["run", *base, "--now", str(now + 1600)]) == 0
        counts = json.loads((queues / "intake.status").read_text(encoding="utf-8"))["counts"]
        assert counts["recovered_assets"] == 1 and counts["assets_finished"] == 1, counts
        with intake.pg_session() as conn:
            assert intake.PgStore(conn).pending_assets() == []
        assert {s for _, s in pg_rows("SELECT drop_path, status FROM intake_file WHERE drop_path LIKE %s", ("%own-s3%",))} == {"archived"}
    finally:
        monkeypatch.delenv(intake.PG_SCHEMA_ENV, raising=False)
        with intake.pg_session() as conn, conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.commit()


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


# =====================================================================================================================
# 1.1.0: review must-fixes
# =====================================================================================================================


def known_from(items: list[dict]) -> intake.Known:
    """What PgStore.load_known returns after earlier (unsealed) passes committed ``items``."""

    known = intake.Known()
    for it in items:
        if it["status"] not in ("admitted", "held", "review", "candidate"):
            continue
        for key in it["identity"]:
            prev = known.identity.get(key)
            known.identity[key] = (min(it["group_id"], prev[0]) if prev else it["group_id"], None)
        known.asset_sha[it["asset_sha256"]] = "intake:earlier"
        if it.get("_signature") is not None and it["status"] in ("admitted", "held"):
            (known.neardup if it["status"] == "admitted" else known.neardup_held).add(it["item_id"], it["_signature"])
    return known


def test_loose_speech_is_owner_memo_with_file_date_session_and_loose_groups_by_file(tmp_path):
    loose = policy("speech/2026-10-15-loose")
    assert loose["owner"] and loose["consent"] == "owner" and loose["speaker"] == "cyril"
    assert loose["group_by"] == "session" and loose["session_from"] == "file_date"
    assert intake.asset_session(loose, "Recording 2026-10-12 0812.m4a", 0.0) == "speech/loose-2026-10-12"
    assert intake.asset_session(loose, "20261013_081203.wav", 0.0) == "speech/loose-2026-10-13"
    assert intake.asset_session(loose, "memo 2026_10_14.wav", 0.0) == "speech/loose-2026-10-14"
    mtime = 1_760_000_000.0
    assert intake.asset_session(loose, "memo.wav", mtime) == f"speech/loose-{intake.utc(mtime)[:10]}"
    third = policy("speech/2026-10-15-loose", {"consent": "third_party_licensed", "licence": "CC BY 4.0"})
    assert not third["owner"] and third["group_by"] == "file"
    own = policy("speech/own-kitchen")
    assert own["session_from"] == "batch" and intake.asset_session(own, "Recording 2026-10-12.m4a", 0.0) == "speech/own-kitchen"
    text_loose = policy("text/2026-10-15-loose", modality_dir="text")
    assert not text_loose["owner"] and text_loose["group_by"] == "file"
    assert policy("text/notes", modality_dir="text")["group_by"] == "batch"
    # loose files are per-file groups (a whole day is not withheld together); a named batch is one group
    holdouts = intake.Holdouts.load(tiny_registry(tmp_path))
    pol = {"text/2026-10-15-loose": text_loose, "text/notes": policy("text/notes", modality_dir="text")}
    a, b = fake_asset("loose-a", "text/2026-10-15-loose", modality="text"), fake_asset("loose-b", "text/2026-10-15-loose", modality="text")
    c, d = fake_asset("notes-c", "text/notes", modality="text"), fake_asset("notes-d", "text/notes", modality="text")
    rows = {x["sha256"]: [{"instruction": "", "response": f"distinct paragraph {x['path']} about rivers and boats"}] for x in (a, b, c, d)}
    items = {it["asset_sha256"]: it for it in intake.plan_items([a, b, c, d], pol, {}, rows, holdouts, intake.Known())}
    assert items[a["sha256"]]["group_id"] != items[b["sha256"]]["group_id"]
    assert items[c["sha256"]]["group_id"] == items[d["sha256"]]["group_id"]
    # a loose owner memo's items carry the file-date session and the session hash split
    memo = fake_asset("memo", "speech/2026-10-15-loose", session_id="speech/loose-2026-10-12")
    it = intake.plan_items([memo], {"speech/2026-10-15-loose": loose}, {}, {}, holdouts, intake.Known())[0]
    assert it["session_id"] == "speech/loose-2026-10-12" and it["owner"] and it["consent"] == "owner"
    assert it["split"] == intake.owner_session_split("speech/loose-2026-10-12") and it["status"] == "candidate"


def test_owner_session_split_is_stable_across_passes_and_never_split(tmp_path):
    holdouts = intake.Holdouts.load(tiny_registry(tmp_path))
    s1, s2, s3, s4 = "speech/own-s1", "speech/own-s2", "speech/own-s3", "speech/own-s4"
    assert intake.owner_session_split(s1) == "train" and intake.owner_session_split(s2) == "test"   # both sides exercised
    pol = {s: policy(s) for s in (s1, s2, s3, s4)}

    def assets_for(session: str, n: int, start: int) -> list[dict]:
        return [fake_asset(f"{session}-{start + i}", session, fp=intake.audio_fingerprint(speechlike(3.0, 700 + start + i)))
                for i in range(n)]

    def transcripts(assets: list[dict]) -> dict:
        return {a["sha256"]: {"transcript": f"owner line {a['path']} read slowly near the window", "tool": "t"} for a in assets}

    p1_assets = assets_for(s1, 3, 0)                                              # pass 1: s1 alone (1.0.0 forced it held)
    p1 = intake.plan_items(p1_assets, pol, transcripts(p1_assets), {}, holdouts, intake.Known())
    p2_assets = assets_for(s1, 2, 10) + assets_for(s2, 2, 20) + assets_for(s3, 2, 30)   # pass 2: s1 again + two new sessions
    p2 = intake.plan_items(p2_assets, pol, transcripts(p2_assets), {}, holdouts, known_from(p1))
    p3_assets = assets_for(s4, 2, 40) + assets_for(s2, 1, 50)                     # pass 3: s4 alone + more s2
    p3 = intake.plan_items(p3_assets, pol, transcripts(p3_assets), {}, holdouts, known_from(p1 + p2))
    sides: dict[str, set] = {}
    for it in p1 + p2 + p3:
        sides.setdefault(it["session_id"], set()).add(it["split"])
        assert it["status"] in ("admitted", "held") and not it["split_conflict"], it["reasons"]
    assert sides == {s: {intake.owner_session_split(s)} for s in (s1, s2, s3, s4)}    # one side each, = the hash
    # replaying pass 2 alone, in another order and with no history, gives the same splits
    again = intake.plan_items(list(reversed(p2_assets)), pol, transcripts(p2_assets), {}, holdouts, intake.Known())
    assert sorted((i["item_id"], i["split"]) for i in again) == sorted((i["item_id"], i["split"]) for i in p2)


def test_owner_session_is_never_split_by_a_held_link(tmp_path):
    holdouts = intake.Holdouts.load(tiny_registry(tmp_path))
    session = "speech/own-s1"                                                        # train by hash
    pol = {session: policy(session)}
    shared = "the lighthouse keeper painted the stairs blue before the storm arrived"
    a = fake_asset("own-linked", session, fp=intake.audio_fingerprint(speechlike(3.0, 801)))
    b = fake_asset("own-plain", session, fp=intake.audio_fingerprint(speechlike(3.0, 802)))
    known = intake.Known()
    known.identity[intake.wording_key(shared)] = ("g" * 64, "test")                 # a sealed HELD item says the same
    trans = {a["sha256"]: {"transcript": shared, "tool": "t"},
             b["sha256"]: {"transcript": "a different owner sentence about the garden gate today", "tool": "t"}}
    by = {it["asset_sha256"]: it for it in intake.plan_items([a, b], pol, trans, {}, holdouts, known)}
    assert by[a["sha256"]]["split"] == "train" and by[a["sha256"]]["status"] == "review"
    assert "split_conflict" in by[a["sha256"]]["reasons"]
    assert by[b["sha256"]]["split"] == "train" and by[b["sha256"]]["status"] == "admitted"   # the session is not dragged
    assert by[a["sha256"]]["group_id"] != by[b["sha256"]]["group_id"]


def test_near_copy_of_held_text_is_refused(tmp_path):
    holdouts = intake.Holdouts.load(tiny_registry(tmp_path))
    base = " ".join(f"gamma{i}" for i in range(150))
    near = base.replace("gamma75", "changed")
    pol = {"text/t": policy("text/t", {"group_by": "file"}, modality_dir="text")}
    a = fake_asset("t-near", "text/t", modality="text")
    rows = {a["sha256"]: [{"instruction": "", "response": near}]}
    known = intake.Known()
    known.neardup_held.add("held-item", intake.minhash_signature(base))              # an earlier pass HELD the original
    it = intake.plan_items([a], pol, {}, rows, holdouts, known)[0]
    assert (it["status"], it["reasons"], it["duplicate_of"]) == ("duplicate_near", ["minhash_held"], "held-item")
    assert intake.plan_items([a], pol, {}, rows, holdouts, intake.Known())[0]["status"] in ("admitted", "held")  # 1.0.0
    # inside one pass: whenever the first of two near-copies is held, the second is refused, not admitted to train
    found = 0
    for k in range(400):
        x, y = fake_asset(f"pair-x-{k}", "text/t", modality="text"), fake_asset(f"pair-y-{k}", "text/t", modality="text")
        body = " ".join(f"w{k}x{i}" for i in range(150))
        prs = {x["sha256"]: [{"instruction": "", "response": body}],
               y["sha256"]: [{"instruction": "", "response": body.replace(f"w{k}x75 ", "zz ")}]}
        items = sorted(intake.plan_items([x, y], pol, {}, prs, holdouts, intake.Known()), key=lambda i: i["item_id"])
        if items[0]["status"] == "held":
            assert items[1]["status"] == "duplicate_near" and items[1]["reasons"] == ["minhash_held"]
            found += 1
            if found == 3:
                break
    assert found == 3


def test_registry_held_minhash_screen_liveness_and_legacy_registry(tmp_path):
    long_held = " ".join(f"delta{i}" for i in range(160))
    reg = tiny_registry(tmp_path, texts=HELD_TEXTS + [long_held])
    holdouts = intake.Holdouts.load(reg)
    assert holdouts.near_source == "registry minhash.npy" and len(holdouts.near.signatures) >= 1
    live = intake.holdouts_liveness(holdouts)
    assert live["live"] and live["near"]["totals"] == {"near_probes": 1, "near_caught": 1}
    rules = {h["rule"] for h in holdouts.screen_text(long_held.replace("delta80 ", "zzz "))}
    assert {"minhash_held", "ngram13"} <= rules
    assert not [h for h in holdouts.screen_text(CLEAN_TEXT) if h["rule"] == "minhash_held"]
    # a registry written by 1.0.0 (no minhash files): signatures are computed at load, same verdicts
    body = json.loads(reg.read_text(encoding="utf-8"))
    body.pop("registry_sha256")
    for name in ("minhash", "minhash_index"):
        body["files"].pop(name)
    intake.dcb.write_json(reg, body, seal_field="registry_sha256")
    legacy = intake.Holdouts.load(reg)
    assert legacy.near_source.startswith("computed at load") and set(legacy.near.signatures) == set(holdouts.near.signatures)
    assert "minhash_held" in {h["rule"] for h in legacy.screen_text(long_held.replace("delta80 ", "zzz "))}


LONG_VAL = "During validation the assistant explained how the harbour pilots coordinate tugboats when fog rolls across the estuary at night."
LONG_TEST = "For the test split the user asked which three migratory birds cross the salt marsh first when the spring tides arrive early."
LONG_TRAIN = "A training conversation described how a baker laminates croissant dough with cold butter folded in three careful turns."
BFCL_Q = "Find the area of a triangle with base ten units and height five units using the geometry calculator function please."
TWIN_Q = "Compute the area of a triangle whose base is ten units and whose height is five units with the geometry calculator tool."
NATIVE_TWIN = "Use the geometry function to compute a triangle area for base ten and height five and report the answer in square units."
OTHER_Q = "Summarise the weekly sales report for the hardware store and list the three best selling products by revenue this week."


def write_registry_v2_fixture(tmp_path: Path) -> dict:
    bank = {"name": "omni_toolcall_bank_v1", "heldout_functions": ["loan_payment", "track_parcel"], "bank_sha256": "f" * 64,
            "cases": [{"id": "tcb-exec_single-000", "lane": "exec_single",
                       "messages": [{"role": "user", "content": "What is the monthly payment on a 25000 euro loan at 6.5 percent over 60 months?"}],
                       "tools": [{"type": "function", "function": {"name": "loan_payment", "description": "Monthly payment, total paid and total interest of a fixed-rate amortizing loan."}},
                                 {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city."}}],
                       "expected": {"calls": []}},
                      {"id": "tcb-missing_tool-000", "lane": "missing_tool", "messages": [{"role": "user", "content": "Turn off the lights in the kitchen."}],
                       "tools": [], "expected": {"no_call": True}}]}
    paths = {"bank": tmp_path / "bank.json", "normalized": tmp_path / "normalized.jsonl", "bfcl": tmp_path / "bfcl62_rids.json",
             "rows": tmp_path / "rows.jsonl"}
    paths["bank"].write_text(json.dumps(bank), encoding="utf-8")
    norm = [{"id": "u-val", "split": "validation", "messages": [{"role": "user", "content": LONG_VAL}, {"role": "assistant", "content": "ok"}],
             "provenance": {"registry_origin": {"rid": "r-val"}}},
            {"id": "u-test", "split": "test", "messages": [{"role": "user", "content": [{"type": "text", "text": LONG_TEST}]}]},
            {"id": "u-train", "split": "train", "messages": [{"role": "user", "content": LONG_TRAIN}]},
            {"id": "u-native-twin", "split": "train", "messages": [{"role": "user", "content": NATIVE_TWIN}]}]
    paths["normalized"].write_bytes(("\n".join(json.dumps(r) for r in norm) + "\n{broken line\n").encode("utf-8"))
    paths["bfcl"].write_text(json.dumps({"rids": ["r-bfcl-1"], "uids": ["u-bfcl-1"], "twin_exposure": {"twin_ids_by_source": {
        "text": ["r-twin-1"], "long": [], "k3": [], "selfgen": [], "native_train": ["u-native-twin"]}}}), encoding="utf-8")
    paths["rows"].write_text("".join(json.dumps(r) + "\n" for r in (
        {"rid": "r-bfcl-1", "instruction": BFCL_Q, "response": "area = 25"}, {"rid": "r-twin-1", "instruction": TWIN_Q, "response": "25"},
        {"rid": "r-other", "instruction": OTHER_Q, "response": "list"})), encoding="utf-8")
    return paths


def test_registry_v2_toolcall_codex_and_bfcl_twins_hits(tmp_path):
    paths = write_registry_v2_fixture(tmp_path)
    specs = [{"name": "omni_toolcall_bank_v1", "kind": "toolcall_bank", "path": str(paths["bank"])},
             {"name": "codex_recovery_heldout", "kind": "codex_normalized_heldout", "path": str(paths["normalized"]),
              "expect_sha256": intake.file_sha256(paths["normalized"])},
             {"name": "bfcl62_twins", "kind": "bfcl62_twins", "path": str(paths["bfcl"]), "rows": str(paths["rows"]),
              "normalized": str(paths["normalized"])}]
    texts, media, ids, banks = [], [], [], []
    for spec in specs:
        entry, t, m, i, f = intake.collect_bank(spec)
        texts += t
        media += m
        ids += i
        banks.append(entry)
    tool, codex, bfcl = banks
    assert tool["heldout_functions"] == ["loan_payment", "track_parcel"] and tool["literals"] == 2
    assert tool["heldout_functions_without_schema"] == ["track_parcel"] and tool["cases"] == 2
    assert codex["rows_by_split"] == {"test": 1, "validation": 1} and {"u-val", "r-val", "u-test"} <= {r["id"] for r in ids}
    assert bfcl["rids"] == 1 and bfcl["twin_rids"] == 1 and bfcl["uids"] == 2
    assert bfcl["rids_resolved_in_rows"] == 2 and bfcl["uids_resolved_in_normalized"] == 1 and bfcl["unresolved"] == ["u-bfcl-1"]
    reg = intake.write_registry(tmp_path / "corpus", 2, texts, media, ids, [], banks, "test")
    holdouts = intake.Holdouts.load(Path(reg["path"]))
    live = intake.holdouts_liveness(holdouts)
    assert live["live"], live
    assert live["text"]["totals"]["literal_caught"] == live["text"]["totals"]["literal_probes"] == 2

    def sources(text: str) -> set:
        return {(h["rule"], h["source"]) for h in holdouts.screen_text(text)}

    assert ("literal", "omni_toolcall_bank_v1") in sources("then the agent called loan_payment(principal=1) again")
    assert ("literal", "omni_toolcall_bank_v1") in sources("track_parcel is a function name too")
    assert any(s == "omni_toolcall_bank_v1" for _, s in sources("Q: What is the monthly payment on a 25000 euro loan at 6.5 percent over 60 months? A: ..."))
    assert ("short_item", "omni_toolcall_bank_v1") in sources("so please turn off the lights in the kitchen right now")
    assert any(s == "codex_recovery_heldout" for _, s in sources("prefix " + LONG_VAL + " suffix"))
    assert any(s == "codex_recovery_heldout" for _, s in sources(LONG_TEST.upper()))
    assert not sources(LONG_TRAIN)                                                     # a train row is not held
    assert any(s == "bfcl62_twins" for _, s in sources(BFCL_Q))
    assert any(s == "bfcl62_twins" for _, s in sources(TWIN_Q))                        # a text twin rid
    assert any(s == "bfcl62_twins" for _, s in sources(NATIVE_TWIN))                   # a native-train twin uid
    assert not sources(OTHER_Q)
    with pytest.raises(intake.Refusal):                                                # pinned bank re-hashed
        intake.collect_bank({**specs[1], "expect_sha256": "0" * 64})
    only_train = tmp_path / "train_only.jsonl"
    only_train.write_text(json.dumps({"id": "x", "split": "train", "messages": []}) + "\n", encoding="utf-8")
    with pytest.raises(intake.Refusal):                                                # a held bank with no held rows
        intake.collect_bank({"name": "c", "kind": "codex_normalized_heldout", "path": str(only_train)})


def make_version(corpus: Path, number: int, rows: list[dict], *, sealed: bool = True) -> Path:
    vdir = corpus / f"v{number}"
    sha, n = intake.write_jsonl_atomic(vdir / "rows" / "text.jsonl", rows)
    body = {"schema": "test", "version": number, "parent_version": number - 1, "files": {"rows/text.jsonl": {"sha256": sha, "rows": n}},
            "holdout_registry": {"version": 1, "sha256": "0" * 64, "path": "x"}, "identity": intake.tool_identity(), "counts": {},
            "row_item_ids_sha256": intake.canonical_sha256(sorted(r["item_id"] for r in rows)), "store_files": {},
            "sealed_utc": intake.utc(), "sealed_by": "test"}
    manifest_sha = intake.dcb.write_json(vdir / "manifest.json", body, seal_field="manifest_sha256")
    if sealed:
        (vdir / "sealed").write_text(json.dumps({"manifest_sha256": manifest_sha}) + "\n", encoding="utf-8")
    return vdir


def test_corpus_held_rows_join_the_next_registry_and_partials_are_never_read(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "v1").mkdir(parents=True)                                               # the adopted v1: no manifest
    held = {"item_id": "i-held", "split": "test", "modality": "text", "asset_sha256": "c" * 64, "store_path": "",
            "instruction": "", "response": LONG_VAL}
    train = {"item_id": "i-train", "split": "train", "modality": "text", "asset_sha256": "d" * 64, "store_path": "",
             "instruction": "", "response": LONG_TRAIN}
    make_version(corpus, 2, [held, train])
    make_version(corpus, 3, [{**held, "item_id": "i-partial", "response": OTHER_Q}], sealed=False)   # a seal that died
    assert intake.sealed_versions(corpus) == [2]
    with pytest.raises(intake.Refusal):
        intake.load_manifest(corpus, 3)
    assert intake.load_manifest(corpus, 3, require_sealed=False)["version"] == 3
    entry, texts, media, ids, fps = intake.collect_bank({"name": "corpus_heldout", "kind": "corpus_versions_heldout",
                                                         "corpus_root": str(corpus)})
    assert [t["text"] for t in texts] == [LONG_VAL] and [m["sha256"] for m in media] == ["c" * 64]
    assert [v["version"] for v in entry["corpus_versions"]] == [2]
    reg = intake.write_registry(corpus, 2, texts + [{"bank": "b", "item": "x", "rule": "text", "text": HELD_TEXTS[0]}], media,
                                ids, fps, [entry], "test")
    holdouts = intake.Holdouts.load(Path(reg["path"]))
    assert holdouts.corpus_versions_covered == [2]
    assert {h["source"] for h in holdouts.screen_text(LONG_VAL)} == {"corpus_heldout"}
    assert not holdouts.screen_text(LONG_TRAIN) and not holdouts.screen_text(OTHER_Q)
    assert holdouts.screen_media("c" * 64) and not holdouts.screen_media("d" * 64)
    moved = intake.move_aside(corpus / "v3", corpus, "test")
    assert not (corpus / "v3").exists() and (moved / "manifest.json").is_file() and moved.parent.name == "_failed_seals"


def test_malformed_transcripts_jsonl_never_raises(tmp_path):
    batch = tmp_path / "b"
    for name in ("a.wav", "b.wav", "c.wav"):
        write_wav(batch / name, speechlike(1.0, 3))
    lines = [json.dumps({"file": "a.wav", "transcript": "good line for a"}).encode(), b"{not json",
             b'{"file": "b.wav", "transcript": "bad \xff\xfe bytes"}', json.dumps(["not", "object"]).encode(),
             json.dumps({"file": "../escape.wav", "transcript": "x"}).encode(), json.dumps({"transcript": "no file"}).encode()]
    (batch / "_transcripts.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    (batch / "c.txt").write_bytes(b"caf\xe9 sidecar in latin-1")
    problems: list[str] = []
    found = intake.sidecar_transcripts(batch, [batch / "a.wav", batch / "b.wav", batch / "c.wav"], problems)
    assert found[str(batch / "a.wav")]["transcript"] == "good line for a"
    assert str(batch / "b.wav") not in found
    assert found[str(batch / "c.wav")]["transcript"].startswith("caf")
    assert "transcripts_jsonl_not_utf8" in problems and "transcripts_jsonl_bad_lines:5" in problems


def test_pdftotext_timeout_is_one_files_problem(tmp_path, monkeypatch):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 broken")
    monkeypatch.setattr(intake.shutil, "which", lambda name: "/usr/bin/" + name)

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pdftotext", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(intake.subprocess, "run", hang)
    assert intake.extract_text_rows(pdf, "pdf") == ([], "unreadable:pdftotext_timeout")
    assert intake.review_reasons({"flags": ["unreadable:pdftotext_timeout"], "modality": "document"}, {}) == ["unreadable:pdftotext_timeout"]


def test_long_recording_keeps_its_sidecar_transcript_as_reference(tmp_path):
    holdouts = intake.Holdouts.load(tiny_registry(tmp_path))
    pol = {"speech/own-s1": policy("speech/own-s1")}
    a = fake_asset("long", "speech/own-s1", duration_s=45.0, segments=[(0.0, 15.0), (15.0, 30.0), (30.0, 45.0)])
    reference = "a long reading of the whole garden list with many words about seeds and soil"
    items = intake.plan_items([a], pol, {a["sha256"]: {"transcript": reference, "tool": "sidecar long.txt"}}, {}, holdouts, intake.Known())
    assert len(items) == 3
    for it in items:
        assert it["transcript_ref"] == reference and "long-form reference" in it["transcript_ref_tool"]
        assert not it.get("transcript") and it["status"] == "candidate" and intake.wording_key(reference) in it["identity"]
    leaked = intake.plan_items([a], pol, {a["sha256"]: {"transcript": HELD_TEXTS[0], "tool": "t"}}, {}, holdouts, intake.Known())
    assert {it["status"] for it in leaked} == {"contaminated"}                         # the reference is screened
