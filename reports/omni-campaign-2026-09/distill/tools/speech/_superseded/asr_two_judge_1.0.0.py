#!/usr/bin/env python3
"""asr_two_judge.py -- two independent ASR judges for speech transcripts and admission (Claude, 2026-10-15).

Design of record: distill/SPEECH_LANE_PLAN_v2_2026-10-14.md section 2.3.3 (transcription by agreement), section 2.1
(admission judges A and B) and the revision note of 2026-10-15 (judge B = Granite Speech 4.1 2B).  The intake lane owns
intake.py; this module is what its `transcribe` step calls (see INTAKE CALL below), and speech_teacher_cache's
`asr-admit` uses admission() the same way.

Judges (both pinned by full revision; a model directory whose fetch receipt names another revision is refused):
  A  Qwen/Qwen3-ASR-0.6B              rev 5eb144179a02acc5e5ba31e748d22b0cf3e303b0  (qwen_asr 0.0.6)
  B  ibm-granite/granite-speech-4.1-2b rev de575db64086f84fdc79da4932d1076e965bc546  (transformers 4.57.6,
                                       GraniteSpeechForConditionalGeneration; no LoRA in 4.1, no peft path)
  Different vendors, encoders and training data, so agreement is evidence and not one model agreeing with itself.  Both
  run in the runtime /mnt/lexar/omni-campaign/runtimes/speech-asr (no pydantic, no ORM: build_speech_runtimes 1.1.0
  gates it), one model per process, A then B, so a card never holds both.

Rules
  agreement(a, b)   no reference text (intake: Cyril's recordings and any dropped speech).  Plan 2.3.3:
                    transcript_conf = 1 - WER(A, B) after normalisation; `auto` when WER <= 5 %, `review` otherwise.
                    WER here is the edit distance over the SHORTER of the two word sequences, i.e. the larger of the
                    two directional WERs, so the verdict does not depend on which judge is called the reference and
                    is never more lenient than the plan's WER(A, B).  Either transcript empty, or a judge error -> review.
  admission(ref, a, b)  synthetic rows with a known text (plan 2.1 / step 9 C-s3): admitted only when BOTH judges equal
                    the reference exactly after normalisation; anything else -> quarantine.
  Normalisation: NFKC, casefold, curly apostrophes folded, hyphens/dashes split words, punctuation dropped, letters and
  digits kept in any script, every CJK/kana/hangul character its own word.  No number normalisation: "20" against
  "twenty" is a disagreement, so formatting differences go to review instead of being guessed away.
  The transcript kept is judge A's text (cased and punctuated): the held speech-in judge is Qwen3-ASR-1.7B, so admitted
  transcripts share its formatting conventions; judge B's text is recorded beside it.

INTAKE CALL (for the intake lane; intake.py 1.0.0 cmd_transcribe still runs one model twice with a 10 % / 0.85 rule)
    sys.path.insert(0, str(HERE / "speech"))
    import asr_two_judge
    with gpu_claim(queues, gpu, "intake-transcribe", wait_s):          # the CALLER holds lock + p40.claims + mem gate
        results = asr_two_judge.transcribe_two_judges(paths_16k_mono, gpu=gpu)   # one dict per path, same order
    for item, res in zip(pending, results):
        transcript, conf = res["transcript"], res["transcript_conf"]
        tstatus = res["transcript_status"]                              # 'auto' | 'review'
        tool = res["transcript_tool"]                                   # both repos@rev + this module's version
        # res["judge_a"]["text"], res["judge_b"]["text"], res["agreement_wer"], res["reasons"] for the review export
  transcribe_two_judges() never takes a GPU claim itself; gpu must be a P40 (1, 2 or 3; GPU5 never) or None for CPU.
  It raises RuntimeError when a judge process fails as a whole; a single unreadable file comes back as `review` with
  reason judge_a_error / judge_b_error.

CLI
    python3 asr_two_judge.py verdict --a TEXT --b TEXT [--reference TEXT]       (any python3; stdlib only)
    python3 asr_two_judge.py transcribe --manifest M.json --out O.json [--gpu N] [--threads 16]
    python3 asr_two_judge.py smoke --librispeech DIR --n 4 --out O.json [--gpu N]  (WER of each judge vs the reference)
    <speech-asr python> -I asr_two_judge.py judge --judge A|B --manifest M --out O --device cpu|cuda:0   (internal)
A manifest is a JSON list of {"id", "audio"[, "reference", "language"]}.  Audio is read at 16 kHz mono.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TOOL = "asr_two_judge"
VERSION = "1.0.0"   # 1.0.0 (2026-10-15): first version (teacher fix (b): judge A Qwen3-ASR-0.6B + judge B Granite 4.1 2B)

MODELS = Path("/mnt/lexar/models")
RUNTIME_PYTHON = "/mnt/lexar/omni-campaign/runtimes/speech-asr/bin/python"
JUDGES: dict[str, dict[str, Any]] = {
    "A": {"repo": "Qwen/Qwen3-ASR-0.6B", "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
          "dir": MODELS / "Qwen/Qwen3-ASR-0.6B", "backend": "qwen_asr"},
    "B": {"repo": "ibm-granite/granite-speech-4.1-2b", "revision": "de575db64086f84fdc79da4932d1076e965bc546",
          "dir": MODELS / "ibm-granite/granite-speech-4.1-2b", "backend": "granite_speech"},
}
AGREEMENT_MAX_WER = 0.05          # plan v2 2.3.3: auto when WER <= 5 %
P40_GPUS = frozenset({1, 2, 3})   # judges are P40 fp32 teachers (plan 2.1); GPU5 never; 3090s / RTX 8000 belong to v1
MEM_GATE_GIB = 40.0               # plan 7.4: every model process gates on MemAvailable
FORBIDDEN_MODULES = ("pydantic", "pydantic_core", "sqlalchemy", "sqlmodel", "peewee", "pony", "tortoise", "django",
                     "mongoengine")
GRANITE_PROMPT = "<|audio|>can you transcribe the speech into a written format?"  # the 4.1 card's raw-ASR prompt
JUDGE_TIMEOUT_S = 6 * 3600

_CJK = "぀-ヿ㐀-䶿一-鿿가-힯豈-﫿"
_WORD_RE = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+(?:'[^\W_{_CJK}]+)*")


# ---------------------------------------------------------------------------------------------------------------------
# rules (stdlib only; what the intake and the tests import)
# ---------------------------------------------------------------------------------------------------------------------


def normalise_words(text: str | None) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[-‐-―]", " ", text)
    return _WORD_RE.findall(text)


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(reference: str | None, hypothesis: str | None) -> float:
    ref, hyp = normalise_words(reference), normalise_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def agreement(text_a: str | None, text_b: str | None) -> dict[str, Any]:
    """Plan 2.3.3 verdict for two judges' transcripts of the same audio (no reference)."""

    wa, wb = normalise_words(text_a), normalise_words(text_b)
    reasons: list[str] = []
    if text_a is None:
        reasons.append("judge_a_error")
    if text_b is None:
        reasons.append("judge_b_error")
    if not wa or not wb:
        if text_a is not None and text_b is not None:
            reasons.append("empty_transcript")
        edits = edit_distance(wa, wb)
        return {"transcript_conf": 0.0, "transcript_status": "review", "agreement_wer": 1.0, "word_edits": edits,
                "words_a": len(wa), "words_b": len(wb), "reasons": reasons}
    edits = edit_distance(wa, wb)
    rate = edits / min(len(wa), len(wb))
    status = "auto" if rate <= AGREEMENT_MAX_WER else "review"
    if status == "review":
        reasons.append("judges_disagree")
    return {"transcript_conf": round(max(0.0, 1.0 - rate), 4), "transcript_status": status,
            "agreement_wer": round(rate, 4), "word_edits": edits, "words_a": len(wa), "words_b": len(wb),
            "reasons": reasons}


def admission(reference: str, text_a: str | None, text_b: str | None) -> dict[str, Any]:
    """Synthetic-row admission (plan 2.1 / C-s3): both judges must reproduce the reference exactly after normalisation."""

    ref = normalise_words(reference)
    a_ok = text_a is not None and normalise_words(text_a) == ref
    b_ok = text_b is not None and normalise_words(text_b) == ref
    return {"admitted": bool(ref) and a_ok and b_ok, "verdict": "admitted" if (ref and a_ok and b_ok) else "quarantine",
            "judge_a_exact": a_ok, "judge_b_exact": b_ok,
            "judge_a_wer": None if text_a is None else round(wer(reference, text_a), 4),
            "judge_b_wer": None if text_b is None else round(wer(reference, text_b), 4)}


def tool_string() -> str:
    a, b = JUDGES["A"], JUDGES["B"]
    return (f"two-judge agreement: {a['repo']}@{a['revision'][:8]} + {b['repo']}@{b['revision'][:8]} "
            f"({TOOL} {VERSION}, WER <= {AGREEMENT_MAX_WER:.0%})")


def combine(paths: Sequence[str], rows_a: Mapping[str, Mapping[str, Any]], rows_b: Mapping[str, Mapping[str, Any]],
            meta_a: Mapping[str, Any], meta_b: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One result per path, in order, from the two judges' per-path rows (keyed by audio path)."""

    out = []
    for path in paths:
        ra, rb = rows_a.get(path) or {}, rows_b.get(path) or {}
        text_a, text_b = ra.get("text"), rb.get("text")
        verdict = agreement(text_a, text_b)
        out.append({"audio": path, "transcript": text_a, **verdict, "transcript_tool": tool_string(),
                    "judge_a": {"repo": meta_a.get("repo"), "revision": meta_a.get("revision"), "text": text_a,
                                "error": ra.get("error"), "seconds": ra.get("seconds")},
                    "judge_b": {"repo": meta_b.get("repo"), "revision": meta_b.get("revision"), "text": text_b,
                                "error": rb.get("error"), "seconds": rb.get("seconds")}})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# orchestration (any python3): one judge process per model, A then B
# ---------------------------------------------------------------------------------------------------------------------


def judge_env(gpu: int | None, threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads)})
    return env


def run_judge(judge: str, manifest: Path, out: Path, *, gpu: int | None, threads: int, runtime_python: str,
              timeout_s: float = JUDGE_TIMEOUT_S) -> dict[str, Any]:
    device = "cpu" if gpu is None else "cuda:0"
    command = [runtime_python, "-I", str(Path(__file__).resolve()), "judge", "--judge", judge, "--manifest",
               str(manifest), "--out", str(out), "--device", device, "--threads", str(threads)]
    proc = subprocess.run(command, env=judge_env(gpu, threads), capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0 or not out.is_file():
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(f"judge {judge} failed (rc {proc.returncode}):\n{tail}")
    return json.loads(out.read_text(encoding="utf-8"))


def transcribe_two_judges(audio_paths: Sequence[str], *, gpu: int | None = None, threads: int = 16,
                          runtime_python: str = RUNTIME_PYTHON, work_dir: str | None = None,
                          language: str = "English") -> list[dict[str, Any]]:
    """Transcribe every path with judge A, then judge B, and return one agreement result per path (same order).

    gpu None = CPU (CUDA hidden); else a P40 index the CALLER has claimed (gpu<N>.lock + p40.claims + MemAvailable)."""

    if gpu is not None and int(gpu) not in P40_GPUS:
        raise ValueError(f"gpu {gpu} refused: the judges run on a P40 (1, 2 or 3) or on CPU; GPU5 never")
    paths = [str(p) for p in audio_paths]
    if not paths:
        return []
    with tempfile.TemporaryDirectory(prefix="asr2j-", dir=work_dir) as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps([{"id": f"{i:06d}", "audio": p, "language": language}
                                        for i, p in enumerate(paths)]), encoding="utf-8")
        res_a = run_judge("A", manifest, root / "a.json", gpu=gpu, threads=threads, runtime_python=runtime_python)
        res_b = run_judge("B", manifest, root / "b.json", gpu=gpu, threads=threads, runtime_python=runtime_python)
    by_a = {r["audio"]: r for r in res_a["rows"]}
    by_b = {r["audio"]: r for r in res_b["rows"]}
    return combine(paths, by_a, by_b, res_a, res_b)


# ---------------------------------------------------------------------------------------------------------------------
# the judge process (runs inside the speech-asr runtime; imports torch)
# ---------------------------------------------------------------------------------------------------------------------


def mem_available_gib() -> float:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 0.0


def check_judge_dir(judge: str) -> Path:
    spec = JUDGES[judge]
    receipt = spec["dir"] / ".speech_fetch_receipt.json"
    if not receipt.is_file():
        raise SystemExit(f"judge {judge}: no fetch receipt in {spec['dir']}")
    body = json.loads(receipt.read_text())
    if body.get("revision") != spec["revision"] or not body.get("complete"):
        raise SystemExit(f"judge {judge}: receipt revision {body.get('revision')} complete={body.get('complete')} "
                         f"!= pinned {spec['revision']}")
    return spec["dir"]


def cmd_judge(args: argparse.Namespace) -> int:
    import importlib.util
    bad = [m for m in FORBIDDEN_MODULES if importlib.util.find_spec(m) is not None]
    if bad:
        print(f"refusing: forbidden modules importable in {sys.executable}: {bad}", file=sys.stderr)
        return 2
    if args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("CPU judge must run with CUDA_VISIBLE_DEVICES=\"\"", file=sys.stderr)
        return 2
    if mem_available_gib() < MEM_GATE_GIB:
        print(f"MemAvailable {mem_available_gib():.1f} GiB < {MEM_GATE_GIB}: refusing", file=sys.stderr)
        return 3
    import librosa
    import torch
    torch.set_num_threads(args.threads)
    model_dir = check_judge_dir(args.judge)
    rows_in = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    started = time.time()
    if JUDGES[args.judge]["backend"] == "qwen_asr":
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(str(model_dir), dtype=torch.float32, device_map=args.device,
                                              max_inference_batch_size=1, max_new_tokens=256)

        def transcribe(wav, language):
            return model.transcribe(audio=(wav, 16000), language=language or None)[0].text
    else:
        from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
        processor = AutoProcessor.from_pretrained(str(model_dir))
        tokenizer = processor.tokenizer
        model, info = GraniteSpeechForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.float32, output_loading_info=True)
        missing = [k for k in info.get("missing_keys", [])]
        if missing:  # a silently re-initialised weight would make judge B a random model
            print(f"refusing: granite load left {len(missing)} weights uninitialised: {missing[:5]}", file=sys.stderr)
            return 4
        model = model.to(args.device).eval()
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": GRANITE_PROMPT}], tokenize=False,
                                               add_generation_prompt=True)

        def transcribe(wav, language):
            inputs = processor(prompt, torch.from_numpy(wav).unsqueeze(0), device=args.device,
                               return_tensors="pt").to(args.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
            new = out[0, inputs["input_ids"].shape[-1]:].unsqueeze(0)
            return tokenizer.batch_decode(new, add_special_tokens=False, skip_special_tokens=True)[0]
    load_seconds = time.time() - started
    rows = []
    for item in rows_in:
        t0 = time.time()
        try:
            wav, _ = librosa.load(item["audio"], sr=16000, mono=True)
            text, error = transcribe(wav, item.get("language", "English")).strip(), None
            audio_seconds = len(wav) / 16000
        except Exception as exc:  # noqa: BLE001 -- one bad file is a review row, not a failed batch
            text, error, audio_seconds = None, repr(exc), None
        seconds = time.time() - t0
        rows.append({"id": item.get("id"), "audio": item["audio"], "text": text, "error": error,
                     "seconds": round(seconds, 2), "audio_seconds": None if audio_seconds is None else round(audio_seconds, 2),
                     "rtf": None if not audio_seconds else round(seconds / audio_seconds, 3)})
        print(f"[judge {args.judge}] {item.get('id')} {seconds:.1f} s: {text if error is None else 'ERROR ' + error}",
              file=sys.stderr, flush=True)
    import resource
    result = {"schema": "omni-campaign.asr-two-judge.judge.v1", "tool": TOOL, "tool_version": VERSION,
              "judge": args.judge, "repo": JUDGES[args.judge]["repo"], "revision": JUDGES[args.judge]["revision"],
              "model_dir": str(model_dir), "device": args.device, "dtype": "float32", "threads": args.threads,
              "load_seconds": round(load_seconds, 1), "rows": rows, "torch": torch.__version__,
              "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.device.startswith("cuda"):
        result["cuda_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# CLI wrappers
# ---------------------------------------------------------------------------------------------------------------------


def cmd_verdict(args: argparse.Namespace) -> int:
    out: dict[str, Any] = {"agreement": agreement(args.a, args.b)}
    if args.reference is not None:
        out["admission"] = admission(args.reference, args.a, args.b)
    print(json.dumps(out, indent=1))
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = transcribe_two_judges([r["audio"] for r in rows], gpu=args.gpu, threads=args.threads)
    for row, res in zip(rows, results):
        res["id"] = row.get("id")
        if row.get("reference"):
            res["admission"] = admission(row["reference"], res["judge_a"]["text"], res["judge_b"]["text"])
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(results), "status": dict(_count(r["transcript_status"] for r in results))}))
    return 0


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def pick_librispeech(root: Path, n: int, salt: str = "chimera-speech-smoke-v1") -> list[dict[str, str]]:
    """N utterances with 8-30 reference words, lowest sha256(salt:utt_id) first (same rule as speech_teacher_smoke)."""

    rows = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            if 8 <= len(text.split()) <= 30:
                key = hashlib.sha256(f"{salt}:{utt}".encode()).hexdigest()
                rows.append((key, utt, text, str(trans.parent / f"{utt}.flac")))
    rows.sort()
    return [{"id": utt, "audio": path, "reference": text, "source": f"librispeech/{root.name}", "pick_key": key}
            for key, utt, text, path in rows[:n]]


def cmd_smoke(args: argparse.Namespace) -> int:
    rows = pick_librispeech(Path(args.librispeech), args.n)
    t0 = time.time()
    results = transcribe_two_judges([r["audio"] for r in rows], gpu=args.gpu, threads=args.threads)
    wall = time.time() - t0
    edits = {"A": 0, "B": 0}
    words = 0
    report = []
    for row, res in zip(rows, results):
        ref = normalise_words(row["reference"])
        words += len(ref)
        line = {"id": row["id"], "reference": row["reference"], "verdict": res["transcript_status"],
                "agreement_wer": res["agreement_wer"], "transcript_conf": res["transcript_conf"]}
        print(f"{row['id']}  REF: {row['reference']}")
        for judge, key in (("A", "judge_a"), ("B", "judge_b")):
            text = res[key]["text"]
            e = edit_distance(ref, normalise_words(text))
            edits[judge] += e
            line[f"judge_{judge.lower()}"] = {"text": text, "word_edits": e, "wer": round(e / max(1, len(ref)), 4),
                                              "seconds": res[key]["seconds"]}
            print(f"   {judge} ({JUDGES[judge]['repo']}): WER {e}/{len(ref)} = {e / max(1, len(ref)):.3f}  {text}")
        line["admission_vs_reference"] = admission(row["reference"], res["judge_a"]["text"], res["judge_b"]["text"])
        print(f"   agreement: {res['transcript_status']} (A/B WER {res['agreement_wer']:.3f}, conf "
              f"{res['transcript_conf']:.3f}); admission vs reference: {line['admission_vs_reference']['verdict']}")
        report.append(line)
    pooled = {j: round(edits[j] / max(1, words), 4) for j in edits}
    summary = {"schema": "omni-campaign.asr-two-judge.smoke.v1", "tool_version": VERSION, "librispeech": args.librispeech,
               "n": len(rows), "device": "cpu" if args.gpu is None else f"gpu{args.gpu}", "threads": args.threads,
               "wall_seconds": round(wall, 1), "pooled_wer": pooled, "reference_words": words,
               "judges": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in JUDGES.items()},
               "rows": report, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"pooled WER vs reference: A {pooled['A']:.4f}, B {pooled['B']:.4f} over {words} words; wall {wall:.0f} s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--a", required=True)
    v.add_argument("--b", required=True)
    v.add_argument("--reference")
    for name in ("transcribe", "smoke"):
        p = sub.add_parser(name)
        if name == "transcribe":
            p.add_argument("--manifest", required=True)
        else:
            p.add_argument("--librispeech", required=True)
            p.add_argument("--n", type=int, default=4)
        p.add_argument("--out", required=True)
        p.add_argument("--gpu", type=int, default=None)
        p.add_argument("--threads", type=int, default=16)
    j = sub.add_parser("judge")
    j.add_argument("--judge", required=True, choices=sorted(JUDGES))
    j.add_argument("--manifest", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--device", default="cpu")
    j.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)
    return {"verdict": cmd_verdict, "transcribe": cmd_transcribe, "smoke": cmd_smoke, "judge": cmd_judge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
