#!/usr/bin/env python3
"""selfgen_sampler.py -- the parent's self-gen corpus sampler (stage C, P40 lane).

WHY THIS FILE EXISTS.  `distill/DISTILL_DESIGN_2026-10-12.md` section 3, row "Self-gen
text": 1.0M tokens free-run SAMPLED from Qwen3.8-27B-Uncensored's Q4_K_M GGUF on
llama-server, one server per P40, no P2P.  Sampling only: the logits are rescored later
in bf16 by `qtext_teacher_cache.py build-text` (wave 2), so what this tool must get right
is the token ids, the prompt they continue, and how each sample ended.  The sampler quant
is ruled by STAGE_D_RULINGS U7 ("the parent's Q4_K_M on the P40s"); the GGUF files are
transient and live in the declared scratch `/mnt/lexar/omni-campaign/scratch/`.

WHAT IS REUSED, NOT RE-IMPLEMENTED.
  * The export is `$OC/export/export_hf_to_gguf.py` (llama.cpp b10752 / b96806d9, built
    for sm_61;75;86 by the export lane), run by `run_selfgen_export_2026-10-13.sh`.
  * The parent-directory digest and the llama-server log parser are the export lane's
    `llama_probe.py`, imported BY PATH (`--export-tools`), never copied: the post-export
    digest must use the same construction as the exporter's pre-export digest or the
    before/after comparison means nothing.
  * The prompt set is the corpus lane's sealed `distill/corpus/selfgen/prompts.jsonl`
    (2,532 prompts, budget 1,000,253 tokens), checked against its manifest sha.
  * The rescoring corpus `export-rescore` writes is read by
    `qtext_teacher_cache.load_text_corpus` -- the test suite loads it through that
    function, so the hand-off is proven against its consumer.

PROMPT IDS.  The prompts are tokenized here with the parent's pinned `tokenizer.json`
(the same construction as distill_corpus_build: `add_special_tokens=False`) and sent to
llama-server as an id list, so the ids the server continues are exactly the ids the
rescoring pass will be teacher-forced on.  A count that differs from the corpus record's
`prompt_tokens` is a refusal.  The server returns the sampled ids (`return_tokens`); an
EOS stop keeps its end-of-generation id as the last token (llama-server appends the id
before it tests for EOG), which is what D2's EOS supervision needs.  llama.cpp also stops
on <|fim_pad|>/<|repo_name|>/<|file_sep|>, which the HF parent samples past; such a row is
labelled `end: server_only_eog`, truncated, and listed in the manifest (1.2.0).

SAMPLING.  The parent's own generation_config.json (temperature 1.0, top_k 20, top_p
0.95), read and locked into the run config; max_new_tokens = the prompt record's
`max_new_tokens` (2,048).  One seed per prompt, sha256(base_seed:rid); with `-np 4`
batching on a GPU the bytes are not bit-reproducible, and the receipt says so.

RUN LAYOUT  <out-dir>/
    run_config.json     locked semantics: prompt file + manifest sha, tokenizer sha,
                        sampling, GGUF identity, shard size, token target.  A resume
                        with different semantics is refused.
    journal.jsonl       one line per completed prompt (append + fsync); resume = replay.
    shard-NNNNN/        `--rows-per-shard` consecutive prompt indexes, written to a temp
                        name and renamed into place only when every row is present:
                          samples.jsonl  one record per prompt (ids, text, stop, timings)
                          tokens.u32     prompt ids + response ids per row, little endian
                          shard.json     rows table, sha256 + bytes per file, checks
    progress.json       rewritten every `--progress-every` s (poll this).
    sessions/*.json     one per `run` session.
    manifest.json       written when the token target is reached and every scheduled
                        prompt has a sealed row; sealed with its own canonical sha.

FAILURES ARE FUNCTIONAL, NUMBERS ARE NOT.  A sampled id outside [0, 248,320), a prompt
whose token count moved, a GGUF whose path or digest is not the locked one, a shard whose
bytes no longer match its record: each stops with a non-zero exit and a record.  Rates,
lengths and truncation counts are diagnostics.

Subcommands
-----------
  tensor-map    parse the converter's --dry-run log against the parent's safetensors
                index: every text tensor mapped, what was skipped (P9 for the parent).
  export-check  re-hash the Q4 GGUF the exporter wrote, re-digest the parent (before ==
                after), and optionally delete the bf16 GGUF from declared scratch.
  serve         launch one llama-server on one allowed card (<= 2,000 MiB in use),
                detached, with a launch receipt and an allocator receipt.
  stop          stop a server this tool launched (by its launch receipt).
  probe         P5: prefill and decode tok/s at 4k, single stream and 4 parallel.
  sanity        sample N prompts, print one line of each, tokenizer + decode parity.
  eog-probe     force each candidate id by logit bias: the live server's end-of-generation
                set must equal EOG_IDS + SERVER_ONLY_EOG_IDS (what `end_labels` assumes).
  run           the resumable sampler.  `--servers-file` (one URL per line) is re-read at
                every progress tick, so a server started later on a freed card joins the
                running session (its /props model_path must be the locked GGUF).
  status        progress + ETA from progress.json.
  verify        re-hash every shard against its record and re-run the structural checks.
  export-rescore  sealed shards -> the safetensors corpus `qtext build-text` reads.

    PY=/home/cereal/model-surgery-lab/venv/bin/python
    T=/home/cereal/omni-campaign/distill/tools
    $PY $T/selfgen_sampler.py serve --gguf <Q4> --gpu-index 1 --port 18311 --name selfgen-gpu1
    $PY $T/selfgen_sampler.py run --server http://127.0.0.1:18311 --server http://127.0.0.1:18312 \\
        --out-dir $OC/distill/selfgen/run-q38-q4km --export-check-receipt <receipt>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

TOOL_NAME = "selfgen_sampler"
TOOL_VERSION = "1.2.0"
RUN_SCHEMA = "omni-distill-selfgen-run-v1"
SHARD_SCHEMA = "omni-distill-selfgen-shard-v1"
MANIFEST_SCHEMA = "omni-distill-selfgen-manifest-v1"
RECEIPT_SCHEMA = "omni-distill-selfgen-receipt-v1"

OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
HERE = Path(__file__).resolve().parent
DEFAULT_PARENT = Path("/home/cereal/models/Qwen3.8-27B-Uncensored")
DEFAULT_EXPORT_TOOLS = OC / "export"
DEFAULT_LLAMACPP_ROOT = OC / "export" / "llamacpp"
DEFAULT_PROMPTS = OC / "distill" / "corpus" / "selfgen" / "prompts.jsonl"
DEFAULT_PROMPTS_MANIFEST = OC / "distill" / "corpus" / "selfgen" / "manifest.json"
DEFAULT_ALLOCATOR_DIR = OC / "composite" / "receipts" / "allocator"
DEFAULT_TPS_DIR = OC / "distill" / "tps"

#: The parent's tokenizer.json, pinned exactly as distill_corpus_build pins it.
QWEN_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
#: Physical text vocabulary (design section 4: every text loss is over [0, 248,320)).
TEXT_VOCAB_ROWS = 248_320
#: <|im_end|>, <|endoftext|> -- the parent's generation_config eos_token_id list.
EOG_IDS = (248_046, 248_044)
#: <|fim_pad|>, <|repo_name|>, <|file_sep|>: llama.cpp b10752 (src/llama-vocab.cpp, the
#: special_fim_pad/rep/sep_id block before the special_eog_ids loop) ALSO ends generation on
#: these, found by their text, while the parent's generation_config treats them as ordinary
#: tokens.  A sample the server stopped on one of them was cut where the HF parent keeps
#: going: it is labelled truncated, never EOS-terminated (`end_labels`).  `eog-probe`
#: checks the live server's EOG set equals EOG_IDS + these.
SERVER_ONLY_EOG_IDS = (248_063, 248_064, 248_065)
#: Their pieces in the pinned tokenizer.json (ordinary added tokens there; asserted by the tests).
SERVER_ONLY_EOG_PIECES = {248_063: "<|fim_pad|>", 248_064: "<|repo_name|>", 248_065: "<|file_sep|>"}
PAD_ID = 248_044
#: Largest card occupancy a launch accepts on its target (workflow GPU rule).
CARD_BUSY_MIB = 2000
#: The P40s this lane may use.  GPU0/4/6 belong to step 1, GPU5 to another project.
DEFAULT_ALLOWED_GPUS = (1, 2, 3)
#: Declared transient scratch (STAGE_D_RULINGS "Transient scratch").  Everything else
#: under /mnt is permanent (CHECKPOINT_POLICY R6) and is never deleted by this tool.
DECLARED_SCRATCH = (Path("/mnt/lexar/omni-campaign/scratch"),)
PROTECTED_PREFIXES = (Path("/mnt"), Path("/home/cereal/models"))
DIGEST_SKIP_NOTE = "llama_probe.directory_digest (export lane), suffix filter recorded"


class SamplerError(RuntimeError):
    """A refusal that is this tool's own, never a bare traceback."""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_bytes_atomic(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with open(tmp, "wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_bytes_atomic(path, (json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    body = {k: v for k, v in payload.items() if k != field}
    body[field] = canonical_sha256(body)
    return body


def check_seal(payload: dict[str, Any], field: str) -> bool:
    body = {k: v for k, v in payload.items() if k != field}
    return payload.get(field) == canonical_sha256(body)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def tool_identity() -> dict[str, Any]:
    me = Path(__file__).resolve()
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(me), "sha256": file_sha256(me),
            "host": platform.node(), "python": platform.python_version()}


def is_under(path: Path, root: Path) -> bool:
    """Path-PARTS containment (so /mnt/lexar_x is not under /mnt/lexar), as oc_checkpoint does."""
    path_parts = Path(os.path.abspath(path)).parts
    root_parts = Path(os.path.abspath(root)).parts
    return path_parts[: len(root_parts)] == root_parts


def scratch_delete_allowed(path: Path, scratch_roots: Iterable[Path]) -> tuple[bool, str]:
    """May this tool `rm` `path`?  Only a .gguf inside a declared scratch root, and never
    anything under /home/cereal/models.  Under /mnt only the declared scratch qualifies."""
    path = Path(os.path.abspath(path))
    if path.suffix != ".gguf":
        return False, f"{path} is not a .gguf file"
    if is_under(path, Path("/home/cereal/models")):
        return False, f"{path} is under the source model library (R6)"
    roots = [Path(os.path.abspath(r)) for r in scratch_roots]
    inside = [r for r in roots if is_under(path, r) and path != r]
    if not inside:
        return False, f"{path} is not inside a declared scratch root {[str(r) for r in roots]}"
    for root in inside:
        if is_under(root, Path("/mnt")) and not any(is_under(root, d) for d in DECLARED_SCRATCH):
            return False, f"scratch root {root} is under /mnt but is not the declared scratch (R6)"
    return True, "inside declared scratch " + str(inside[0])


# --------------------------------------------------------------------------
# The export lane's llama_probe.  Imported by path, never copied.
# --------------------------------------------------------------------------

_PROBE: dict[str, Any] = {}


def llama_probe(export_tools: Path) -> Any:
    key = str(Path(export_tools).resolve())
    if key in _PROBE:
        return _PROBE[key]
    path = Path(export_tools) / "llama_probe.py"
    if not path.is_file():
        raise SamplerError(f"the export lane's llama_probe.py is missing at {path}")
    spec = importlib.util.spec_from_file_location(f"llama_probe_{len(_PROBE)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PROBE[key] = module
    return module


def probe_identity(export_tools: Path) -> dict[str, Any]:
    path = Path(export_tools) / "llama_probe.py"
    return {"path": str(path), "sha256": file_sha256(path)}


# --------------------------------------------------------------------------
# Tokenizer: the parent's pinned tokenizer.json, as the corpus lane uses it.
# --------------------------------------------------------------------------


class ParentTokenizer:
    def __init__(self, parent_dir: Path, allow_unpinned: bool = False) -> None:
        from tokenizers import Tokenizer

        path = Path(parent_dir) / "tokenizer.json"
        if not path.is_file():
            raise SamplerError(f"no tokenizer.json in {parent_dir}")
        self.path = path
        self.sha256 = file_sha256(path)
        if self.sha256 != QWEN_TOKENIZER_SHA256 and not allow_unpinned:
            raise SamplerError(f"{path} hashes {self.sha256}; the pinned 27B tokenizer is {QWEN_TOKENIZER_SHA256}")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.defined_rows = int(self.tokenizer.get_vocab_size(with_added_tokens=True))
        for eog in EOG_IDS:
            if self.tokenizer.id_to_token(eog) is None:
                raise SamplerError(f"end-of-generation id {eog} is not defined by {path}")

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)

    def piece(self, token_id: int) -> str | None:
        return self.tokenizer.id_to_token(int(token_id))

    def report(self) -> dict[str, Any]:
        return {"tokenizer_json": str(self.path), "sha256": self.sha256, "defined_rows": self.defined_rows,
                "add_special_tokens": False, "eog_ids": list(EOG_IDS),
                "eog_pieces": [self.piece(i) for i in EOG_IDS]}


# --------------------------------------------------------------------------
# Prompts: the corpus lane's sealed self-gen set.
# --------------------------------------------------------------------------

PROMPT_FIELDS = ("rid", "route", "prompt_text", "prompt_tokens", "max_new_tokens")


def load_prompts(path: Path, manifest_path: Path | None) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise SamplerError(f"prompt file {path} does not exist")
    sha = file_sha256(path)
    manifest_info: dict[str, Any] = {"path": None}
    if manifest_path is not None:
        manifest = read_json(Path(manifest_path))
        expected = ((manifest.get("file") or {}).get("sha256"))
        if expected != sha:
            raise SamplerError(f"prompt file {path} hashes {sha}; its manifest {manifest_path} records {expected}")
        manifest_info = {"path": str(Path(manifest_path).resolve()), "sha256": file_sha256(Path(manifest_path)),
                         "manifest_sha256": manifest.get("manifest_sha256"),
                         "prompts": manifest.get("prompts"), "budget_tokens": manifest.get("budget_tokens"),
                         "budget_target": manifest.get("budget_target"),
                         "post_screen": {k: (manifest.get("post_screen") or {}).get(k) for k in ("units_checked", "units_hit")}}
    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [f for f in PROMPT_FIELDS if f not in row]
            if missing:
                raise SamplerError(f"{path}:{line_no + 1} is missing {missing}")
            prompts.append(row)
    rids = [p["rid"] for p in prompts]
    if len(set(rids)) != len(rids):
        raise SamplerError(f"{path} carries duplicate rids")
    if manifest_path is not None and manifest_info.get("prompts") not in (None, len(prompts)):
        raise SamplerError(f"{path} has {len(prompts)} prompts; the manifest says {manifest_info['prompts']}")
    return {"path": str(path), "sha256": sha, "prompts": prompts, "manifest": manifest_info}


def prompt_seed(base_seed: int, rid: str) -> int:
    return int(hashlib.sha256(f"{int(base_seed)}:{rid}".encode("utf-8")).hexdigest()[:8], 16)


def encode_prompt(tok: ParentTokenizer, prompt: dict[str, Any]) -> list[int]:
    ids = tok.encode(prompt["prompt_text"])
    if len(ids) != int(prompt["prompt_tokens"]):
        raise SamplerError(f"prompt {prompt['rid']} tokenizes to {len(ids)} ids; the corpus record says {prompt['prompt_tokens']}")
    return ids


# --------------------------------------------------------------------------
# Sampling configuration: the parent's own generation_config.json.
# --------------------------------------------------------------------------


def parent_sampling(parent_dir: Path) -> dict[str, Any]:
    path = Path(parent_dir) / "generation_config.json"
    config = read_json(path)
    eos = config.get("eos_token_id")
    eos_ids = sorted(int(e) for e in (eos if isinstance(eos, list) else [eos]))
    if eos_ids != sorted(EOG_IDS):
        raise SamplerError(f"{path} eos_token_id {eos_ids} differs from the locked EOG ids {sorted(EOG_IDS)}")
    return {
        "source": str(path),
        "source_sha256": file_sha256(path),
        "do_sample": bool(config.get("do_sample", True)),
        "temperature": float(config.get("temperature", 1.0)),
        "top_k": int(config.get("top_k", 0) or 0),
        "top_p": float(config.get("top_p", 1.0)),
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }


def completion_payload(prompt_ids: list[int], n_predict: int, seed: int, sampling: dict[str, Any],
                       ignore_eos: bool = False) -> dict[str, Any]:
    return {
        "prompt": [int(i) for i in prompt_ids],
        "n_predict": int(n_predict),
        "temperature": sampling["temperature"],
        "top_k": sampling["top_k"],
        "top_p": sampling["top_p"],
        "min_p": sampling["min_p"],
        "typical_p": sampling["typical_p"],
        "repeat_penalty": sampling["repeat_penalty"],
        "presence_penalty": sampling["presence_penalty"],
        "frequency_penalty": sampling["frequency_penalty"],
        "seed": int(seed),
        "cache_prompt": False,
        "return_tokens": True,
        "ignore_eos": bool(ignore_eos),
        "stream": False,
    }


# --------------------------------------------------------------------------
# llama-server HTTP (stdlib only).
# --------------------------------------------------------------------------


def http_json(url: str, payload: Any | None = None, timeout: float = 60.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def server_props(url: str) -> dict[str, Any]:
    props = http_json(url.rstrip("/") + "/props", timeout=30)
    settings = props.get("default_generation_settings") or {}
    return {"model_path": props.get("model_path"), "total_slots": props.get("total_slots"),
            "build_info": props.get("build_info"), "n_ctx": settings.get("n_ctx", props.get("n_ctx")),
            "chat_template_present": bool(props.get("chat_template"))}


def server_healthy(url: str) -> bool:
    try:
        return http_json(url.rstrip("/") + "/health", timeout=10).get("status") == "ok"
    except Exception:  # noqa: BLE001 - a probe; the caller decides
        return False


def server_tokenize(url: str, text: str) -> list[int]:
    body = http_json(url.rstrip("/") + "/tokenize",
                     {"content": text, "add_special": False, "parse_special": True}, timeout=120)
    return [int(t if isinstance(t, int) else t.get("id")) for t in body.get("tokens", [])]


def complete(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.time()
    body = http_json(url.rstrip("/") + "/completion", payload, timeout=timeout)
    return body, time.time() - started


# --------------------------------------------------------------------------
# One sample: the structural checks every record carries.
# --------------------------------------------------------------------------


def build_record(index: int, prompt: dict[str, Any], prompt_ids: list[int], seed: int, body: dict[str, Any],
                 wall_s: float, server: str, attempt: int, tok: ParentTokenizer) -> dict[str, Any]:
    tokens = body.get("tokens")
    if not isinstance(tokens, list):
        raise SamplerError(f"server {server} returned no token ids for prompt {prompt['rid']} (return_tokens unsupported?)")
    ids = [int(t) for t in tokens]
    bad = [t for t in ids if t < 0 or t >= TEXT_VOCAB_ROWS]
    if bad:
        raise SamplerError(f"prompt {prompt['rid']}: sampled ids outside [0, {TEXT_VOCAB_ROWS}): {bad[:5]}")
    if not ids:
        raise SamplerError(f"prompt {prompt['rid']}: the server returned an empty sample")
    stop_type = str(body.get("stop_type") or ("eos" if body.get("stopped_eos") else "limit" if body.get("stopped_limit") else "unknown"))
    labels = end_labels(stop_type, ids)
    eog_terminated = labels["eog_terminated"]
    body_ids = ids[:-1] if eog_terminated else ids
    # HF semantics: a server-only EOG id is an ordinary token, so its piece stays in `text`
    text = tok.decode(body_ids)
    content = body.get("content")
    timings = body.get("timings") or {}
    record = {
        "index": int(index),
        "rid": prompt["rid"],
        "uid": prompt.get("uid"),
        "record_id": prompt.get("record_id"),
        "dataset_id": prompt.get("dataset_id"),
        "route": prompt.get("route"),
        "seed": int(seed),
        "prompt_tokens": len(prompt_ids),
        "prompt_ids_sha256": bytes_sha256(struct.pack(f"<{len(prompt_ids)}I", *prompt_ids)),
        "response_ids": ids,
        "response_tokens": len(ids),
        "stop_type": stop_type,
        **labels,
        "undefined_ids": sum(1 for t in ids if t >= tok.defined_rows),
        "text": text,
        "text_matches_server": server_text_matches(content, text, ids, labels, tok),
        "server": server,
        "attempt": int(attempt),
        "wall_s": round(float(wall_s), 3),
        "timings": {k: timings.get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second")},
        "completed_utc": utc_now(),
    }
    if content != text:
        record["server_content"] = content
    return record


def end_labels(stop_type: str, ids: list[int]) -> dict[str, Any]:
    """How a sample ended, from its stop type and ids alone (so 1.1.0 records relabel).

    end = "eog"             last id is an HF end-of-generation id (EOS supervision)
          "server_only_eog" the server stopped on <|fim_pad|>/<|repo_name|>/<|file_sep|>:
                            the HF parent would have continued -> truncated
          "limit"           max_new_tokens reached -> truncated
          other             any other stop without an HF EOG id -> truncated
    `truncated` is exactly "no HF EOG id at the end": D2 never gives such a row an EOS."""
    last = int(ids[-1]) if ids else None
    eog = last in EOG_IDS
    if eog:
        end = "eog"
    elif stop_type == "eos" and last in SERVER_ONLY_EOG_IDS:
        end = "server_only_eog"
    elif stop_type == "limit":
        end = "limit"
    else:
        end = f"{stop_type}_without_eog"
    return {"end": end, "eog_terminated": eog, "eog_id": last if eog else None,
            "server_only_eog_id": last if end == "server_only_eog" else None, "truncated": not eog}


def server_text_matches(content: Any, text: str, ids: list[int], labels: dict[str, Any], tok: ParentTokenizer) -> bool:
    """Decode parity.  On a server-only EOG stop llama-server may leave the stop token's
    piece out of `content` while HF keeps it as an ordinary token; either form matches."""
    if content == text:
        return True
    return labels["end"] == "server_only_eog" and content == tok.decode(ids[:-1])


def relabel(record: dict[str, Any]) -> dict[str, Any]:
    """A journal record with its end labels recomputed by `end_labels` (idempotent)."""
    out = dict(record)
    out.update(end_labels(record["stop_type"], record["response_ids"]))
    if out["end"] == "server_only_eog" and not out.get("text_matches_server") and "server_content" in out:
        # a 1.1.0 record: content omitted the server-only stop piece; the text (HF decode
        # of every id, the piece included) is unchanged, so this is recomputable exactly
        piece_free = out["text"]
        stop_piece = SERVER_ONLY_EOG_PIECES.get(out["server_only_eog_id"], "")
        if stop_piece and piece_free.endswith(stop_piece):
            piece_free = piece_free[: -len(stop_piece)]
        out["text_matches_server"] = out["server_content"] == piece_free
    return out


def record_anomalies(record: dict[str, Any]) -> list[str]:
    """Diagnostics a record can carry; none of them stops the run."""
    out = []
    if record.get("end", "") == "server_only_eog":
        out.append("server_only_eog_stop")
    elif record["stop_type"] == "eos" and not record["eog_terminated"]:
        out.append("eos_stop_without_eog_id")
    if record["eog_terminated"] and record["stop_type"] not in ("eos",):
        out.append("eog_id_without_eos_stop")
    if record["undefined_ids"]:
        out.append("undefined_ids")
    if not record["text_matches_server"]:
        out.append("text_differs_from_server")
    return out


# --------------------------------------------------------------------------
# Shards.
# --------------------------------------------------------------------------


def shard_name(shard: int) -> str:
    return f"shard-{shard:05d}"


def write_shard(out_dir: Path, shard: int, records: list[dict[str, Any]], prompt_ids_by_index: dict[int, list[int]],
                index_range: tuple[int, int]) -> dict[str, Any]:
    """Seal one shard: temp directory, every file hashed, renamed into place.  End labels
    are recomputed from the ids (`relabel`), so journal lines written by 1.1.0 seal right."""
    records = sorted((relabel(r) for r in records), key=lambda r: r["index"])
    expected = list(range(index_range[0], index_range[1]))
    if [r["index"] for r in records] != expected:
        raise SamplerError(f"{shard_name(shard)}: rows {[r['index'] for r in records][:8]}... are not exactly {index_range}")
    final = out_dir / shard_name(shard)
    if final.exists():
        raise SamplerError(f"{final} already exists")
    tmp = out_dir / f".{shard_name(shard)}.tmp-{os.getpid()}"
    if tmp.exists():
        for child in tmp.iterdir():
            child.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    blob = bytearray()
    lines = []
    for record in records:
        prompt_ids = prompt_ids_by_index[record["index"]]
        if bytes_sha256(struct.pack(f"<{len(prompt_ids)}I", *prompt_ids)) != record["prompt_ids_sha256"]:
            raise SamplerError(f"row {record['index']} ({record['rid']}): prompt ids moved since sampling")
        ids = list(prompt_ids) + list(record["response_ids"])
        rows.append({"index": record["index"], "rid": record["rid"], "offset": len(blob) // 4,
                     "prompt_len": len(prompt_ids), "response_len": len(record["response_ids"]),
                     "stop_type": record["stop_type"], "eog_terminated": record["eog_terminated"]})
        blob += struct.pack(f"<{len(ids)}I", *ids)
        lines.append(json.dumps(record, sort_keys=True, ensure_ascii=False))
    samples = ("\n".join(lines) + "\n").encode("utf-8")
    files = {"samples.jsonl": samples, "tokens.u32": bytes(blob)}
    for name, data in files.items():
        with open(tmp / name, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    shard_record = seal({
        "schema": SHARD_SCHEMA,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "shard": shard,
        "index_range": list(index_range),
        "rows": rows,
        "row_count": len(rows),
        "response_tokens": sum(r["response_len"] for r in rows),
        "prompt_tokens": sum(r["prompt_len"] for r in rows),
        "files": {name: {"bytes": len(data), "sha256": bytes_sha256(data)} for name, data in files.items()},
        "checks": shard_checks(records),
        "sealed_utc": utc_now(),
    }, "shard_sha256")
    with open(tmp / "shard.json", "w", encoding="utf-8") as handle:
        json.dump(shard_record, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(tmp)
    os.replace(tmp, final)
    fsync_directory(out_dir)
    return shard_record


def shard_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    records = [relabel(r) for r in records]
    anomalies: dict[str, int] = {}
    for record in records:
        for name in record_anomalies(record):
            anomalies[name] = anomalies.get(name, 0) + 1
    stops: dict[str, int] = {}
    for record in records:
        stops[record["stop_type"]] = stops.get(record["stop_type"], 0) + 1
    return {
        "ids_in_text_vocab": all(0 <= t < TEXT_VOCAB_ROWS for r in records for t in r["response_ids"]),
        "stop_types": dict(sorted(stops.items())),
        "eog_terminated": sum(1 for r in records if r["eog_terminated"]),
        "truncated": sum(1 for r in records if r["truncated"]),
        "server_only_eog": sum(1 for r in records if r["end"] == "server_only_eog"),
        "anomalies": dict(sorted(anomalies.items())),
    }


def read_shard(shard_dir: Path, verify_bytes: bool = True) -> dict[str, Any]:
    record = read_json(shard_dir / "shard.json")
    if not check_seal(record, "shard_sha256"):
        raise SamplerError(f"{shard_dir}/shard.json does not match its own seal")
    data = {}
    for name, info in record["files"].items():
        blob = (shard_dir / name).read_bytes()
        if verify_bytes and (len(blob) != info["bytes"] or bytes_sha256(blob) != info["sha256"]):
            raise SamplerError(f"{shard_dir}/{name} no longer matches its record")
        data[name] = blob
    samples = [json.loads(line) for line in data["samples.jsonl"].decode("utf-8").splitlines() if line.strip()]
    tokens = struct.unpack(f"<{len(data['tokens.u32']) // 4}I", data["tokens.u32"])
    return {"record": record, "samples": samples, "tokens": tokens}


def check_shard_structure(shard: dict[str, Any]) -> list[str]:
    """Structural re-check of a sealed shard: ids, offsets, rows vs samples."""
    problems = []
    record, samples, tokens = shard["record"], shard["samples"], shard["tokens"]
    if len(samples) != record["row_count"]:
        problems.append(f"{len(samples)} samples vs row_count {record['row_count']}")
    cursor = 0
    for row, sample in zip(record["rows"], samples):
        if row["index"] != sample["index"] or row["rid"] != sample["rid"]:
            problems.append(f"row {row['index']} does not match sample {sample['index']}")
        if row["offset"] != cursor:
            problems.append(f"row {row['index']} offset {row['offset']} != {cursor}")
        length = row["prompt_len"] + row["response_len"]
        response = list(tokens[row["offset"] + row["prompt_len"]: row["offset"] + length])
        if response != list(sample["response_ids"]):
            problems.append(f"row {row['index']} response ids differ between tokens.u32 and samples.jsonl")
        prompt = tokens[row["offset"]: row["offset"] + row["prompt_len"]]
        if bytes_sha256(struct.pack(f"<{len(prompt)}I", *prompt)) != sample["prompt_ids_sha256"]:
            problems.append(f"row {row['index']} prompt ids do not hash to the sampled prompt")
        if any(t >= TEXT_VOCAB_ROWS for t in response):
            problems.append(f"row {row['index']} has ids outside the text vocab")
        cursor += length
    if cursor != len(tokens):
        problems.append(f"rows cover {cursor} ids; tokens.u32 holds {len(tokens)}")
    return problems


# --------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------


class JournalState:
    """Completed records by prompt index, replayed from journal.jsonl.

    A kill mid-write (SIGKILL, OOM, ENOSPC, power) leaves a torn last line with no
    newline.  It is cut off the file BEFORE the file is reopened for append: otherwise the
    next record would be glued onto the fragment and lost on the following replay, and a
    shard sealed from it would make every later resume refuse.  The cut is byte-level, so
    a tail torn inside a multi-byte character is handled too.  The file is read and
    written as bytes; one complete line = one record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[int, dict[str, Any]] = {}
        self.dropped_lines = 0
        self.dropped_tail_bytes = 0
        self.duplicates = 0
        self.broken: str | None = None
        if path.is_file():
            data = path.read_bytes()
            if data and not data.endswith(b"\n"):
                keep = data.rfind(b"\n") + 1
                self.dropped_tail_bytes = len(data) - keep
                self.dropped_lines += 1
                with open(path, "r+b") as handle:
                    handle.truncate(keep)
                    handle.flush()
                    os.fsync(handle.fileno())
                data = data[:keep]
            for line in data.split(b"\n"):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line.decode("utf-8"))
                except ValueError:
                    # a complete but unreadable line (not a torn tail); the prompt is re-sampled
                    self.dropped_lines += 1
                    continue
                if record["index"] in self.records:
                    self.duplicates += 1
                    continue
                self.records[record["index"]] = record
        self.handle = open(path, "ab", buffering=0)
        self.lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> bool:
        """Append + fsync one record.  A failed write is cut back to the last complete line
        and the journal refuses further appends this session (SamplerError), so nothing is
        ever glued onto a fragment; the prompts not journaled are re-sampled on resume."""
        with self.lock:
            if self.broken:
                raise SamplerError(f"journal {self.path} is closed for this session: {self.broken}")
            if record["index"] in self.records:
                self.duplicates += 1
                return False
            line = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            fd = self.handle.fileno()
            size = os.fstat(fd).st_size
            try:
                # unbuffered: a failed write leaves no buffered remainder to be flushed later
                view = memoryview(line)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            except OSError as exc:
                self.broken = f"append of index {record['index']} failed: {exc!r}"
                try:
                    os.ftruncate(fd, size)
                    os.fsync(fd)
                except OSError as cut_exc:
                    self.broken += f"; cut back failed: {cut_exc!r} (the next session cuts the torn tail)"
                raise SamplerError(f"journal {self.path}: {self.broken}") from exc
            self.records[record["index"]] = record
            return True

    def close(self) -> None:
        self.handle.close()


def locked_config(args: argparse.Namespace, prompts: dict[str, Any], tok: ParentTokenizer,
                  sampling: dict[str, Any], gguf: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "prompts": {"path": prompts["path"], "sha256": prompts["sha256"], "count": len(prompts["prompts"]),
                    "manifest": prompts["manifest"]},
        "tokenizer": tok.report(),
        "sampling": sampling,
        "base_seed": int(args.base_seed),
        "max_new_tokens_rule": "each prompt record's max_new_tokens",
        "rows_per_shard": int(args.rows_per_shard),
        "token_target": int(args.token_target),
        "token_count_rule": "response ids including the terminating end-of-generation id",
        "text_vocab_rows": TEXT_VOCAB_ROWS,
        "eog_ids": list(EOG_IDS),
        "gguf": gguf,
        "reproducibility": "one seed per prompt (sha256(base_seed:rid)); llama-server batches up to -np slots per "
                           "card, so samples are not bit-reproducible across runs; ids + seed + server are recorded",
    }


def gguf_identity(args: argparse.Namespace) -> dict[str, Any]:
    if args.export_check_receipt:
        receipt = read_json(Path(args.export_check_receipt))
        q4 = (receipt.get("q4") or {})
        if not q4.get("path") or not q4.get("sha256"):
            raise SamplerError(f"{args.export_check_receipt} carries no Q4 path + sha256")
        return {"path": q4["path"], "sha256": q4["sha256"], "bytes": q4.get("bytes"),
                "source": {"export_check_receipt": str(Path(args.export_check_receipt).resolve()),
                           "receipt_sha256": file_sha256(Path(args.export_check_receipt))},
                "parent_directory_sha256": (receipt.get("parent_digest_after") or {}).get("directory_sha256")}
    if args.gguf:
        return {"path": os.path.abspath(args.gguf), "sha256": args.gguf_sha256, "bytes": None,
                "source": {"declared": True}, "parent_directory_sha256": None}
    raise SamplerError("run needs --export-check-receipt (or --gguf for a fixture)")


def check_servers(urls: list[str], gguf: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for url in urls:
        if not server_healthy(url):
            raise SamplerError(f"server {url} is not healthy")
        props = server_props(url)
        model_path = props.get("model_path")
        if model_path and os.path.abspath(model_path) != os.path.abspath(gguf["path"]):
            raise SamplerError(f"server {url} serves {model_path}; the locked GGUF is {gguf['path']}")
        out[url] = props
    return out


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = Path(args.out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.session = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"

    # ---------------------------------------------------------------- setup
    def prepare(self) -> None:
        args = self.args
        self.lock_handle = open(self.out_dir / "run.lock", "a+")
        try:
            import fcntl

            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # pragma: no cover - the server is Linux
            pass
        except OSError as exc:
            raise SamplerError(f"another run holds {self.out_dir}/run.lock") from exc
        if (self.out_dir / "manifest.json").exists():
            raise SamplerError(f"{self.out_dir} is finalized (manifest.json exists); use a new --out-dir")
        self.prompts = load_prompts(Path(args.prompts), Path(args.prompts_manifest) if args.prompts_manifest else None)
        self.tok = ParentTokenizer(Path(args.parent_dir), allow_unpinned=args.allow_unpinned_tokenizer)
        self.sampling = parent_sampling(Path(args.parent_dir))
        self.gguf = gguf_identity(args)
        config = locked_config(args, self.prompts, self.tok, self.sampling, self.gguf)
        config_path = self.out_dir / "run_config.json"
        if config_path.exists():
            previous = read_json(config_path)
            if not check_seal(previous, "config_sha256"):
                raise SamplerError(f"{config_path} does not match its own seal")
            prev_body = {k: v for k, v in previous.items() if k not in ("config_sha256", "created_utc", "tool")}
            if canonical_sha256(prev_body) != canonical_sha256(config):
                diff = sorted(k for k in set(prev_body) | set(config) if prev_body.get(k) != config.get(k))
                raise SamplerError(f"resume refused: locked semantics differ in {diff}")
            self.config = previous
        else:
            self.config = seal(dict(config, created_utc=utc_now(), tool=tool_identity()), "config_sha256")
            write_json_atomic(config_path, self.config)
        self.prompt_list = self.prompts["prompts"]
        self.n = len(self.prompt_list)
        self.rows_per_shard = int(args.rows_per_shard)
        self.target = int(args.token_target)
        self.journal = JournalState(self.out_dir / "journal.jsonl")
        self.sealed = {int(p.name.split("-")[1]) for p in self.out_dir.glob("shard-*") if p.is_dir()}
        for shard in sorted(self.sealed):
            # a shard is sealed FROM the journal, so its rows must all be there; otherwise a
            # resume would re-sample rows that already have sealed bytes
            loaded = read_shard(self.out_dir / shard_name(shard))
            lo, hi = loaded["record"]["index_range"]
            missing = [i for i in range(lo, hi) if i not in self.journal.records]
            if missing:
                raise SamplerError(f"{shard_name(shard)} is sealed but rows {missing[:5]} are not in the journal")
        self.prompt_ids: dict[int, list[int]] = {}
        self.inflight: set[int] = set()
        self.cutoff: int | None = None
        self.servers = list(args.server or [])
        self.server_props = check_servers(self.servers, self.gguf)
        self.server_stats = {u: {"done": 0, "tokens": 0, "errors": 0, "consecutive_errors": 0, "dead": False}
                             for u in self.servers}
        self.servers_file_skips: dict[str, str] = {}
        self.refresh_servers(start=False)
        if not self.servers:
            raise SamplerError("no healthy server: pass --server or list one in --servers-file")
        self.session_started = time.time()
        self.session_tokens_at_start = self.done_tokens()
        self.rate_window: list[tuple[float, int]] = [(time.time(), self.session_tokens_at_start)]
        self.errors: list[dict[str, Any]] = []
        self.failure: str | None = None
        log(f"run {self.out_dir}: {self.n} prompts, target {self.target:,} tokens, "
            f"{len(self.journal.records)} already sampled ({self.done_tokens():,} tokens), "
            f"{len(self.sealed)} shards sealed, servers {self.servers}")
        if self.journal.dropped_lines:
            log(f"journal: dropped {self.journal.dropped_lines} torn line(s) (torn tail cut: "
                f"{self.journal.dropped_tail_bytes} bytes); those prompts are re-sampled")

    def refresh_servers(self, start: bool = True) -> list[str]:
        """Join every healthy URL newly listed in --servers-file; return the ones added."""
        path = self.args.servers_file
        if not path or not Path(path).is_file():
            return []
        added = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            url = line.strip()
            if not url or url.startswith("#") or url in self.server_stats:
                continue
            try:
                props = check_servers([url], self.gguf)[url]
            except SamplerError as exc:
                if self.servers_file_skips.get(url) != str(exc):
                    log(f"servers file: {url} not joined yet: {exc}")
                self.servers_file_skips[url] = str(exc)
                continue
            with self.state_lock:
                self.servers.append(url)
                self.server_props[url] = props
                self.server_stats[url] = {"done": 0, "tokens": 0, "errors": 0, "consecutive_errors": 0, "dead": False,
                                          "joined_utc": utc_now()}
            added.append(url)
            log(f"servers file: joined {url} {props}")
            if start:
                self.start_workers(url)
        return added

    def start_workers(self, url: str) -> None:
        for slot in range(int(self.args.slots_per_server)):
            thread = threading.Thread(target=self.worker, args=(url,), name=f"w-{url}-{slot}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def done_tokens(self) -> int:
        return sum(r["response_tokens"] for r in self.journal.records.values())

    def ids_for(self, index: int) -> list[int]:
        if index not in self.prompt_ids:
            self.prompt_ids[index] = encode_prompt(self.tok, self.prompt_list[index])
        return self.prompt_ids[index]

    # ---------------------------------------------------------------- scheduling
    def take(self) -> int | None:
        """The lowest prompt index that is neither sampled nor in flight, below the cutoff.

        Lowest-first keeps the sampled set a prefix: holes left by a killed session or a
        failed request are filled before the frontier moves, so the cutoff (fixed the
        moment the token target is reached) always has every index below it sampled."""
        with self.state_lock:
            if self.stop_event.is_set():
                return None
            if self.cutoff is None and self.done_tokens() >= self.target:
                self.cutoff = max(list(self.journal.records) + list(self.inflight) + [-1]) + 1
                log(f"token target reached: {self.done_tokens():,} >= {self.target:,}; scheduling cut at index {self.cutoff}")
            limit = self.n if self.cutoff is None else self.cutoff
            index = 0
            while index < limit and (index in self.journal.records or index in self.inflight):
                index += 1
            if index >= limit:
                return None
            self.inflight.add(index)
            return index

    def release(self, index: int) -> None:
        with self.state_lock:
            self.inflight.discard(index)

    # ---------------------------------------------------------------- workers
    def worker(self, url: str) -> None:
        args = self.args
        stats = self.server_stats[url]
        while not self.stop_event.is_set() and not stats["dead"]:
            index = self.take()
            if index is None:
                return
            prompt = self.prompt_list[index]
            try:
                prompt_ids = self.ids_for(index)
            except SamplerError as exc:
                self.fail(str(exc))
                self.release(index)
                return
            seed = prompt_seed(args.base_seed, prompt["rid"])
            payload = completion_payload(prompt_ids, int(prompt["max_new_tokens"]), seed, self.sampling)
            record = None
            for attempt in range(1, int(args.retries) + 2):
                if self.stop_event.is_set():
                    break
                try:
                    body, wall = complete(url, payload, timeout=float(args.request_timeout))
                    record = build_record(index, prompt, prompt_ids, seed, body, wall, url, attempt, self.tok)
                    break
                except SamplerError as exc:
                    self.fail(str(exc))
                    break
                except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError, ValueError, OSError) as exc:
                    with self.state_lock:
                        stats["errors"] += 1
                        stats["consecutive_errors"] += 1
                        self.errors.append({"utc": utc_now(), "server": url, "index": index, "attempt": attempt, "error": repr(exc)[:300]})
                        self.errors = self.errors[-50:]
                        if stats["consecutive_errors"] >= int(args.max_consecutive_errors):
                            stats["dead"] = True
                    log(f"{url} index {index} attempt {attempt}: {exc!r}"[:300])
                    if stats["dead"]:
                        log(f"{url}: {stats['consecutive_errors']} consecutive errors, worker set stops")
                        break
                    self.stop_event.wait(min(60.0, float(args.retry_backoff) * attempt))
            if record is None:
                self.release(index)
                continue
            try:
                appended = self.journal.append(record)
            except SamplerError as exc:
                # disk full / IO error: stop; the journal was cut back to its last whole line
                self.fail(str(exc))
                self.release(index)
                return
            if appended:
                with self.state_lock:
                    stats["done"] += 1
                    stats["tokens"] += record["response_tokens"]
                    stats["consecutive_errors"] = 0
            self.release(index)
            self.maybe_seal(index // self.rows_per_shard)
            if args.stop_after and len(self.journal.records) >= int(args.stop_after):
                # test hook: a clean kill point for the resume tests
                self.stop_event.set()

    def fail(self, message: str) -> None:
        with self.state_lock:
            if self.failure is None:
                self.failure = message
        log(f"FUNCTIONAL FAILURE: {message}")
        self.stop_event.set()

    # ---------------------------------------------------------------- sealing
    def shard_range(self, shard: int, final: bool = False) -> tuple[int, int] | None:
        lo = shard * self.rows_per_shard
        hi = min((shard + 1) * self.rows_per_shard, self.n)
        if final and self.cutoff is not None:
            hi = min(hi, self.cutoff)
        return (lo, hi) if hi > lo else None

    def maybe_seal(self, shard: int, final: bool = False) -> None:
        with self.state_lock:
            if shard in self.sealed:
                return
            span = self.shard_range(shard, final=final)
            if span is None:
                return
            if any(i not in self.journal.records for i in range(*span)):
                return
            records = [self.journal.records[i] for i in range(*span)]
            self.sealed.add(shard)
        try:
            ids = {i: self.ids_for(i) for i in range(*span)}
            info = write_shard(self.out_dir, shard, records, ids, span)
        except SamplerError as exc:
            with self.state_lock:
                self.sealed.discard(shard)
            self.fail(str(exc))
            return
        log(f"sealed {shard_name(shard)} rows {span[0]}..{span[1] - 1}: {info['response_tokens']:,} tokens, "
            f"stops {info['checks']['stop_types']}, sha {info['shard_sha256'][:12]}")

    # ---------------------------------------------------------------- progress
    def progress(self, final: bool = False) -> dict[str, Any]:
        now = time.time()
        with self.state_lock:
            done_tokens = self.done_tokens()
            self.rate_window.append((now, done_tokens))
            horizon = now - float(self.args.rate_window_s)
            while len(self.rate_window) > 2 and self.rate_window[1][0] < horizon:
                self.rate_window.pop(0)
            t0, k0 = self.rate_window[0]
            recent = (done_tokens - k0) / (now - t0) if now > t0 else 0.0
            session_rate = (done_tokens - self.session_tokens_at_start) / max(1e-9, now - self.session_started)
            remaining = max(0, self.target - done_tokens)
            rate = recent if recent > 0 else session_rate
            body = {
                "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
                "updated_utc": utc_now(),
                "session": self.session,
                "out_dir": str(self.out_dir),
                "prompts_total": self.n,
                "prompts_done": len(self.journal.records),
                "inflight": sorted(self.inflight),
                "response_tokens_done": done_tokens,
                "token_target": self.target,
                "fraction_of_target": round(done_tokens / self.target, 5) if self.target else None,
                "rate_tokens_per_s_recent": round(recent, 3),
                "rate_window_s": float(self.args.rate_window_s),
                "rate_tokens_per_s_session": round(session_rate, 3),
                "eta_s": round(remaining / rate, 1) if rate > 0 else None,
                "eta_utc": (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + remaining / rate)) if rate > 0 else None),
                "shards_sealed": len(self.sealed),
                "cutoff_index": self.cutoff,
                "servers": self.server_stats,
                "recent_errors": self.errors[-10:],
                "failure": self.failure,
                "final": final,
            }
        write_json_atomic(self.out_dir / "progress.json", body)
        return body

    # ---------------------------------------------------------------- main loop
    def execute(self) -> int:
        """Run one session; the run lock and the journal handle are always released."""
        self.lock_handle = None
        self.journal = None
        try:
            return self._execute()
        finally:
            if self.journal is not None and not self.journal.handle.closed:
                self.journal.close()
            if self.lock_handle is not None:
                self.lock_handle.close()

    def _execute(self) -> int:
        self.prepare()
        session_path = self.out_dir / "sessions" / f"{self.session}.json"
        session = {"session": self.session, "started_utc": utc_now(), "argv": sys.argv, "tool": tool_identity(),
                   "servers": self.server_props, "pid": os.getpid(),
                   "resumed_with": {"records": len(self.journal.records), "shards_sealed": len(self.sealed),
                                    "dropped_journal_lines": self.journal.dropped_lines,
                                    "dropped_journal_tail_bytes": self.journal.dropped_tail_bytes}}
        write_json_atomic(session_path, session)

        def on_signal(signum: int, _frame: Any) -> None:
            log(f"signal {signum}: finishing in-flight requests is skipped; the journal keeps every completed sample")
            self.stop_event.set()

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        # every shard already complete in the journal is sealed first (a kill between the
        # journal append and the seal leaves exactly that state)
        for shard in range((self.n + self.rows_per_shard - 1) // self.rows_per_shard):
            self.maybe_seal(shard)
        self.threads: list[threading.Thread] = []
        for url in list(self.servers):
            self.start_workers(url)
        threads = self.threads
        last = 0.0
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
            if time.time() - last >= float(self.args.progress_every):
                last = time.time()
                if not self.stop_event.is_set():
                    self.refresh_servers()
                p = self.progress()
                log(f"progress {p['prompts_done']}/{self.n} prompts, {p['response_tokens_done']:,}/{self.target:,} tokens, "
                    f"{p['rate_tokens_per_s_recent']} tok/s recent, eta {p['eta_utc']}")
        for thread in threads:
            thread.join()
        finished = self.failure is None and not self.stop_event.is_set()
        if finished and self.cutoff is None and len(self.journal.records) == self.n:
            self.cutoff = self.n
        if finished and self.cutoff is not None:
            for shard in range((self.cutoff + self.rows_per_shard - 1) // self.rows_per_shard):
                self.maybe_seal(shard, final=True)
        complete_run = (finished and self.cutoff is not None and self.failure is None
                        and all(i in self.journal.records for i in range(self.cutoff)))
        progress = self.progress(final=complete_run)
        session.update({"ended_utc": utc_now(), "failure": self.failure, "complete": complete_run,
                        "records_at_end": len(self.journal.records), "tokens_at_end": progress["response_tokens_done"]})
        write_json_atomic(session_path, session)
        if self.failure:
            log(f"STOPPED on a functional failure: {self.failure}")
            return 1
        if not complete_run:
            dead = [u for u, s in self.server_stats.items() if s["dead"]]
            log(f"stopped before completion (dead servers {dead}); re-run the same command to resume")
            return 2
        manifest = finalize(self.out_dir, self.config, self.cutoff, self.rows_per_shard)
        log(f"manifest {self.out_dir / 'manifest.json'} sha {manifest['manifest_sha256'][:16]}: "
            f"{manifest['totals']['rows']} rows, {manifest['totals']['response_tokens']:,} tokens")
        return 0


def finalize(out_dir: Path, config: dict[str, Any], cutoff: int, rows_per_shard: int) -> dict[str, Any]:
    """End labels, stop types and anomalies are recomputed from every sample's ids
    (`relabel`), not summed from shard.json checks: shards sealed by 1.1.0 carry no
    server-only-EOG class, and the manifest must list those rows whichever tool sealed them."""
    shards = []
    totals = {"rows": 0, "response_tokens": 0, "prompt_tokens": 0, "eog_terminated": 0, "truncated": 0,
              "server_only_eog": 0}
    ends: dict[str, int] = {}
    server_only_rows: list[dict[str, Any]] = []
    stops: dict[str, int] = {}
    anomalies: dict[str, int] = {}
    by_route: dict[str, dict[str, int]] = {}
    for shard in range((cutoff + rows_per_shard - 1) // rows_per_shard):
        loaded = read_shard(out_dir / shard_name(shard))
        problems = check_shard_structure(loaded)
        if problems:
            raise SamplerError(f"{shard_name(shard)}: {problems[:3]}")
        rec = loaded["record"]
        shards.append({"name": shard_name(shard), "index_range": rec["index_range"], "rows": rec["row_count"],
                       "response_tokens": rec["response_tokens"], "shard_sha256": rec["shard_sha256"],
                       "files": rec["files"]})
        totals["rows"] += rec["row_count"]
        totals["response_tokens"] += rec["response_tokens"]
        totals["prompt_tokens"] += rec["prompt_tokens"]
        for sample in map(relabel, loaded["samples"]):
            totals["eog_terminated"] += int(sample["eog_terminated"])
            totals["truncated"] += int(sample["truncated"])
            stops[sample["stop_type"]] = stops.get(sample["stop_type"], 0) + 1
            ends[sample["end"]] = ends.get(sample["end"], 0) + 1
            for k in record_anomalies(sample):
                anomalies[k] = anomalies.get(k, 0) + 1
            if sample["end"] == "server_only_eog":
                totals["server_only_eog"] += 1
                server_only_rows.append({"index": sample["index"], "rid": sample["rid"], "shard": shard_name(shard),
                                         "stop_id": sample["server_only_eog_id"],
                                         "response_tokens": sample["response_tokens"]})
            route = by_route.setdefault(sample.get("route") or "?", {"rows": 0, "response_tokens": 0})
            route["rows"] += 1
            route["response_tokens"] += sample["response_tokens"]
    if totals["rows"] != cutoff:
        raise SamplerError(f"shards hold {totals['rows']} rows; the cutoff is {cutoff}")
    sessions = sorted(p.name for p in (out_dir / "sessions").glob("*.json"))
    manifest = seal({
        "schema": MANIFEST_SCHEMA,
        "tool": tool_identity(),
        "design": "DISTILL_DESIGN_2026-10-12 section 3 'Self-gen text'; STAGE_D_RULINGS U7",
        "run_config_sha256": config["config_sha256"],
        "prompts": config["prompts"],
        "gguf": config["gguf"],
        "sampling": config["sampling"],
        "cutoff_index": cutoff,
        "cutoff_rule": "prompts are scheduled in file order; scheduling stops once the sampled response tokens reach "
                       "the target; every prompt below the cutoff is sampled",
        "shard_read_order": [s["name"] for s in shards],
        "shards": shards,
        "totals": totals,
        "stop_types": dict(sorted(stops.items())),
        "ends": dict(sorted(ends.items())),
        "end_rule": "eog = last id in the parent's generation_config eos ids (EOS supervision); every other row is "
                    "truncated, incl. server_only_eog (llama.cpp also stops on <|fim_pad|>/<|repo_name|>/<|file_sep|>, "
                    "which the HF parent samples past); labels recomputed from the ids",
        "server_only_eog_rows": server_only_rows,
        "anomalies": dict(sorted(anomalies.items())),
        "by_route": dict(sorted(by_route.items())),
        "sessions": sessions,
        "finalized_utc": utc_now(),
        "logits": "none stored: rescored in bf16 by qtext_teacher_cache build-text (export-rescore writes its corpus)",
    }, "manifest_sha256")
    write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------
# verify / status / export-rescore
# --------------------------------------------------------------------------


def verify_run(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    names = sorted(p.name for p in out_dir.glob("shard-*") if p.is_dir())
    manifest = None
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if not check_seal(manifest, "manifest_sha256"):
            log("FAIL manifest.json does not match its own seal")
            return 1
        if names != manifest["shard_read_order"]:
            log(f"FAIL shard set on disk {names[:3]}... differs from the manifest")
            return 1
    failures = 0
    tokens = 0
    for name in names:
        try:
            loaded = read_shard(out_dir / name)
        except SamplerError as exc:
            log(f"FAIL {name}: {exc}")
            failures += 1
            continue
        problems = check_shard_structure(loaded)
        if manifest is not None:
            entry = next(s for s in manifest["shards"] if s["name"] == name)
            if entry["shard_sha256"] != loaded["record"]["shard_sha256"]:
                problems.append("shard_sha256 differs from the manifest")
        if problems:
            log(f"FAIL {name}: {problems[:3]}")
            failures += 1
        tokens += loaded["record"]["response_tokens"]
    log(f"verify {out_dir}: {len(names)} shards, {tokens:,} response tokens, {failures} failing, "
        f"manifest {'sealed ' + manifest['manifest_sha256'][:16] if manifest else 'absent (run not finalized)'}")
    return 1 if failures else 0


def status(out_dir: Path) -> int:
    path = Path(out_dir) / "progress.json"
    if not path.exists():
        log(f"no progress.json in {out_dir}")
        return 1
    p = read_json(path)
    print(json.dumps({k: p[k] for k in ("updated_utc", "prompts_done", "prompts_total", "response_tokens_done",
                                         "token_target", "fraction_of_target", "rate_tokens_per_s_recent",
                                         "rate_tokens_per_s_session", "eta_utc", "shards_sealed", "failure", "final")},
                     indent=1))
    for url, s in p["servers"].items():
        print(f"  {url}: done {s['done']} tokens {s['tokens']:,} errors {s['errors']} dead {s['dead']}")
    return 0


def export_rescore(out_dir: Path, dest: Path) -> dict[str, Any]:
    """Sealed shards -> `input_ids` [N, L] int32, `lengths` [N], `store_mask` [N, L],
    `eog_terminated` [N] uint8.

    Teacher-forced position t predicts token t+1 (qtext_teacher_cache), so the stored
    positions are [prompt_len - 1, length - 2]: one distribution per sampled token,
    the terminating EOG id included.  `eog_terminated` (and the sidecar's `row_end`) is
    recomputed from the ids: 1 only when the row ends on a parent EOS id; a row the server
    cut on a server-only EOG id is 0 (truncated), so D2 gives it no EOS supervision."""
    import numpy as np
    from safetensors.numpy import save_file

    out_dir = Path(out_dir)
    manifest = read_json(out_dir / "manifest.json")
    if not check_seal(manifest, "manifest_sha256"):
        raise SamplerError("manifest.json does not match its own seal")
    rows = []
    for name in manifest["shard_read_order"]:
        loaded = read_shard(out_dir / name)
        if check_shard_structure(loaded):
            raise SamplerError(f"{name} fails its structural check")
        for row, sample in zip(loaded["record"]["rows"], map(relabel, loaded["samples"])):
            ids = loaded["tokens"][row["offset"]: row["offset"] + row["prompt_len"] + row["response_len"]]
            rows.append((dict(row, end=sample["end"], eog_terminated=sample["eog_terminated"]), ids))
    width = max(len(ids) for _, ids in rows)
    input_ids = np.full((len(rows), width), PAD_ID, dtype=np.int32)
    lengths = np.zeros((len(rows),), dtype=np.int64)
    store = np.zeros((len(rows), width), dtype=np.uint8)
    eog = np.asarray([1 if r["eog_terminated"] else 0 for r, _ in rows], dtype=np.uint8)
    for n, (row, ids) in enumerate(rows):
        input_ids[n, : len(ids)] = np.asarray(ids, dtype=np.int64).astype(np.int32)
        lengths[n] = len(ids)
        store[n, row["prompt_len"] - 1: len(ids) - 1] = 1
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    save_file({"input_ids": input_ids, "lengths": lengths, "store_mask": store, "eog_terminated": eog}, str(tmp),
              metadata={"source_manifest_sha256": manifest["manifest_sha256"], "tool": f"{TOOL_NAME} {TOOL_VERSION}"})
    os.replace(tmp, dest)
    sidecar = seal({
        "schema": RECEIPT_SCHEMA + "/rescore-corpus",
        "tool": tool_identity(),
        "corpus": {"path": str(dest.resolve()), "sha256": file_sha256(dest), "bytes": dest.stat().st_size},
        "source_manifest_sha256": manifest["manifest_sha256"],
        "rows": len(rows),
        "row_length": width,
        "stored_positions": int(store.sum()),
        "response_tokens": int(sum(r["response_len"] for r, _ in rows)),
        "store_rule": "store_mask[n, t] = 1 for t in [prompt_len - 1, length - 2]",
        "pad_id": PAD_ID,
        "eog_terminated_rows": int(eog.sum()),
        "truncated_rows": int(len(rows) - eog.sum()),
        "end_rule": "eog_terminated[n] = 1 iff the row's last id is a parent EOS id; server_only_eog rows are truncated",
        "row_ids": [r["rid"] for r, _ in rows],
        "row_index": [r["index"] for r, _ in rows],
        "row_end": [r["end"] for r, _ in rows],
    }, "receipt_sha256")
    write_json_atomic(dest.with_suffix(".rows.json"), sidecar)
    return sidecar


# --------------------------------------------------------------------------
# tensor-map / export-check
# --------------------------------------------------------------------------

_RE_MAP = re.compile(r"^INFO:hf-to-gguf:(?P<gguf>[A-Za-z0-9_.]+),\s+torch\.(?P<dtype>\w+)\s+-->\s+(?P<qtype>\w+),\s+shape\s+=\s+\{(?P<shape>[^}]*)\}")
_RE_SKIP_HINT = re.compile(r"(skip|Skip|ignor)")


def tensor_map(log_path: Path, parent_dir: Path) -> dict[str, Any]:
    """The converter's dry-run tensor list vs the parent's safetensors index."""
    index = read_json(Path(parent_dir) / "model.safetensors.index.json")
    hf_names = sorted(index["weight_map"])
    classes: dict[str, int] = {}
    for name in hf_names:
        if ".visual." in name or name.startswith("model.visual") or name.startswith("visual."):
            key = "visual"
        elif name.startswith("mtp.") or ".mtp." in name:
            key = "mtp"
        elif "language_model" in name or name.startswith("lm_head") or name.startswith("model."):
            key = "text"
        else:
            key = "other"
        classes[key] = classes.get(key, 0) + 1
    mapped = []
    other_lines = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            m = _RE_MAP.match(line.strip())
            if m:
                mapped.append({"gguf": m.group("gguf"), "from": m.group("dtype"), "to": m.group("qtype"),
                               "shape": [int(x) for x in m.group("shape").replace(" ", "").split(",") if x]})
            elif _RE_SKIP_HINT.search(line) or "ERROR" in line or "Traceback" in line:
                other_lines.append(line.rstrip()[:300])
    blocks = sorted({int(m["gguf"].split(".")[1]) for m in mapped if m["gguf"].startswith("blk.")})
    per_block: dict[int, int] = {}
    for m in mapped:
        if m["gguf"].startswith("blk."):
            b = int(m["gguf"].split(".")[1])
            per_block[b] = per_block.get(b, 0) + 1
    to_types: dict[str, int] = {}
    for m in mapped:
        to_types[m["to"]] = to_types.get(m["to"], 0) + 1
    return {
        "log": str(log_path),
        "log_sha256": file_sha256(Path(log_path)),
        "hf_index": str(Path(parent_dir) / "model.safetensors.index.json"),
        "hf_tensor_count": len(hf_names),
        "hf_tensor_classes": dict(sorted(classes.items())),
        "gguf_tensor_count": len(mapped),
        "gguf_non_block": sorted(m["gguf"] for m in mapped if not m["gguf"].startswith("blk.")),
        "gguf_blocks": [blocks[0], blocks[-1]] if blocks else None,
        "gguf_block_count": len(blocks),
        "gguf_tensors_per_block": dict(sorted({str(k): v for k, v in per_block.items()}.items(), key=lambda kv: int(kv[0]))),
        "gguf_out_types": dict(sorted(to_types.items())),
        "converter_notes": other_lines[:40],
        "mapped": mapped,
    }


def export_check(args: argparse.Namespace) -> dict[str, Any]:
    probe = llama_probe(Path(args.export_tools))
    receipt_path = Path(args.export_receipt)
    receipt = read_json(receipt_path)
    artifacts = receipt.get("artifacts") or {}
    q4 = artifacts.get(args.quant)
    full = artifacts.get(receipt.get("outtype") or "bf16")
    if not q4:
        raise SamplerError(f"{receipt_path} records no {args.quant} artifact")
    stages = {s["stage"]: s for s in receipt.get("stages", [])}
    bad = [name for name, s in stages.items() if s.get("rc") not in (0,)]
    if bad:
        raise SamplerError(f"export stages failed: {bad}")
    q4_path = Path(q4["path"])
    q4_sha = file_sha256(q4_path)
    if q4_sha != q4["sha256"] or q4_path.stat().st_size != q4["bytes"]:
        raise SamplerError(f"{q4_path} no longer matches the export receipt")
    before = receipt.get("source_directory_digest") or {}
    if not before.get("directory_sha256"):
        raise SamplerError("the export receipt carries no parent digest (was it run with --skip-digest?)")
    started = time.time()
    after = probe.directory_digest(str(args.parent_dir))
    digest_wall = round(time.time() - started, 2)
    parent_equal = after["directory_sha256"] == before["directory_sha256"]
    out = {
        "schema": RECEIPT_SCHEMA + "/export-check",
        "tool": tool_identity(),
        "llama_probe": probe_identity(Path(args.export_tools)),
        "export_receipt": {"path": str(receipt_path.resolve()), "sha256": file_sha256(receipt_path)},
        "llama_cpp_commit": receipt.get("llama_cpp_commit"),
        "build_receipt_sha256": receipt.get("build_receipt_sha256"),
        "hf_config": {k: (receipt.get("hf_config") or {}).get(k) for k in ("architectures", "model_type", "num_hidden_layers", "hidden_size", "vocab_size", "config_sha256")},
        "q4": {"quant": args.quant, "path": str(q4_path), "bytes": q4_path.stat().st_size, "sha256": q4_sha,
               "rehashed_equal_to_receipt": True},
        "parent_dir": str(args.parent_dir),
        "parent_digest_before": {k: before.get(k) for k in ("directory_sha256", "file_count", "suffix_filter")},
        "parent_digest_after": {k: after.get(k) for k in ("directory_sha256", "file_count", "suffix_filter")},
        "parent_digest_after_wall_s": digest_wall,
        "parent_unchanged": parent_equal,
        "stage_walls_s": {name: s.get("wall_s") for name, s in stages.items()},
        "full_gguf": None,
        "checked_utc": utc_now(),
    }
    if not parent_equal:
        raise SamplerError(f"the parent's digest moved during the export: {before.get('directory_sha256')} -> {after['directory_sha256']}")
    if full:
        full_path = Path(full["path"])
        entry = {"path": str(full_path), "bytes": full.get("bytes"), "sha256": full.get("sha256"), "deleted": False}
        if args.delete_full:
            allowed, why = scratch_delete_allowed(full_path, [Path(r) for r in args.scratch_root])
            if not allowed:
                raise SamplerError(f"refusing to delete {full_path}: {why}")
            if full_path.exists():
                os.remove(full_path)
                entry.update({"deleted": True, "deleted_utc": utc_now(), "delete_rule": why})
            else:
                entry.update({"deleted": None, "note": "already absent"})
        out["full_gguf"] = entry
    return seal(out, "receipt_sha256")


# --------------------------------------------------------------------------
# serve / stop
# --------------------------------------------------------------------------


def gpu_table() -> list[dict[str, Any]]:
    rc = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    if rc.returncode != 0:
        raise SamplerError(f"nvidia-smi failed: {rc.stderr.strip()[:200]}")
    return parse_gpu_table(rc.stdout)


def parse_gpu_table(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2],
                     "memory_used_mib": int(float(parts[3])), "memory_total_mib": int(float(parts[4]))})
    return rows


def check_card(table: list[dict[str, Any]], index: int, allowed: Iterable[int], busy_mib: int = CARD_BUSY_MIB) -> dict[str, Any]:
    allowed = [int(a) for a in allowed]
    if int(index) not in allowed:
        raise SamplerError(f"GPU{index} is not in this lane's allowed cards {allowed}")
    card = next((c for c in table if c["index"] == int(index)), None)
    if card is None:
        raise SamplerError(f"GPU{index} is not present")
    if card["memory_used_mib"] > busy_mib:
        raise SamplerError(f"GPU{index} has {card['memory_used_mib']} MiB in use (> {busy_mib}); not starting")
    return card


def serve_command(args: argparse.Namespace) -> list[str]:
    root = Path(args.llamacpp_root)
    return [str(root / "bin" / "llama-server"), "-m", str(args.gguf), "-ngl", "999",
            "-c", str(int(args.slots) * int(args.ctx_per_slot)), "-np", str(int(args.slots)),
            "--host", "127.0.0.1", "--port", str(int(args.port)), "-t", str(int(args.threads)),
            "-fa", args.flash_attn, "-ctk", args.kv_type, "-ctv", args.kv_type] + list(args.extra or [])


def serve(args: argparse.Namespace) -> dict[str, Any]:
    table = gpu_table()
    card = check_card(table, args.gpu_index, args.allowed_gpus)
    if not Path(args.gguf).is_file():
        raise SamplerError(f"{args.gguf} does not exist")
    if server_healthy(f"http://127.0.0.1:{args.port}"):
        raise SamplerError(f"port {args.port} already answers /health")
    log_path = Path(args.log_dir) / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = serve_command(args)
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = card["uuid"]
    env["LD_LIBRARY_PATH"] = str(Path(args.llamacpp_root) / "bin") + ":" + env.get("LD_LIBRARY_PATH", "")
    with open(log_path, "ab") as handle:
        proc = subprocess.Popen(["nice", "-n", str(int(args.nice))] + cmd, stdout=handle, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    started = time.time()
    url = f"http://127.0.0.1:{args.port}"
    healthy = False
    while time.time() - started < float(args.health_timeout):
        if proc.poll() is not None:
            break
        if server_healthy(url):
            healthy = True
            break
        time.sleep(3)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    probe = llama_probe(Path(args.export_tools))
    analysis = probe.analyse_server_log(text)
    receipt = {
        "schema": RECEIPT_SCHEMA + "/serve",
        "tool": tool_identity(),
        "name": args.name,
        "url": url,
        "pid": proc.pid,
        "command": cmd,
        "nice": int(args.nice),
        "env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": card["uuid"]},
        "card_before": card,
        "gguf": str(Path(args.gguf).resolve()),
        "log": str(log_path),
        "healthy": healthy,
        "health_wait_s": round(time.time() - started, 1),
        "exited": proc.poll(),
        "props": server_props(url) if healthy else None,
        "log_analysis": {k: analysis.get(k) for k in ("flash_attn_state", "kv_type_k", "kv_type_v", "n_ctx", "n_ctx_slot",
                                                       "n_slots", "offloaded", "kv_buffer_mib", "fallback_evidence")
                         if k in analysis},
        "started_utc": utc_now(),
    }
    receipt_path = Path(args.receipt_dir) / f"serve-{args.name}.json"
    write_json_atomic(receipt_path, seal(receipt, "receipt_sha256"))
    allocator = {
        "name": args.name,
        "kind": "host-llama-server",
        "gpus": [card["index"]],
        "gpu_uuids": [card["uuid"]],
        "pid": proc.pid,
        "launcher": f"{Path(__file__).resolve()} serve ({TOOL_NAME} {TOOL_VERSION})",
        "launcher_sha256": file_sha256(Path(__file__).resolve()),
        "log": str(log_path),
        "owner": "claude",
        "purpose": args.purpose,
        "server": "ai",
        "started_utc": receipt["started_utc"],
        "state": "running" if healthy else "failed to become healthy",
        "card_memory_used_mib_at_start": card["memory_used_mib"],
    }
    write_json_atomic(Path(args.allocator_dir) / f"{args.name}.json", allocator)
    if not healthy:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        raise SamplerError(f"llama-server {args.name} did not become healthy (exit {proc.poll()}); log {log_path}")
    return receipt


def stop_server(args: argparse.Namespace) -> dict[str, Any]:
    receipt = read_json(Path(args.launch_receipt))
    pid = int(receipt["pid"])
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    result = {"pid": pid, "name": receipt["name"], "stopped_utc": utc_now()}
    if cmdline_path.exists():
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        if "llama-server" not in cmdline or receipt["gguf"] not in cmdline:
            raise SamplerError(f"pid {pid} is not the llama-server this receipt launched: {cmdline[:200]}")
        os.killpg(pid, signal.SIGTERM)
        for _ in range(60):
            if not cmdline_path.exists():
                break
            time.sleep(1)
        result["state"] = "stopped" if not cmdline_path.exists() else "still running after 60 s"
    else:
        result["state"] = "already exited"
    alloc_path = Path(args.allocator_dir) / f"{receipt['name']}.json"
    if alloc_path.exists():
        alloc = read_json(alloc_path)
        alloc.update({"state": f"released ({result['state']})", "ended_utc": result["stopped_utc"]})
        if args.outcome:
            alloc["outcome"] = args.outcome
        write_json_atomic(alloc_path, alloc)
    return result


# --------------------------------------------------------------------------
# probe (P5) / sanity
# --------------------------------------------------------------------------


class CardSampler(threading.Thread):
    """Whole-card nvidia-smi memory peak while a probe runs."""

    def __init__(self, index: int | None, period: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.index = index
        self.period = period
        self.peak = None
        self.halt = threading.Event()

    def run(self) -> None:
        while not self.halt.is_set() and self.index is not None:
            try:
                card = next(c for c in gpu_table() if c["index"] == self.index)
                self.peak = max(self.peak or 0, card["memory_used_mib"])
            except Exception:  # noqa: BLE001 - a sampler; a missed sample is not a failure
                pass
            self.halt.wait(self.period)


def filler_ids(tok: ParentTokenizer, prompts: list[dict[str, Any]], target: int) -> list[int]:
    """A ~`target`-token chat prompt made of real self-gen prompt text (no synthetic filler)."""
    head = tok.encode("<|im_start|>user\n")
    tail = tok.encode("\nSummarize the material above in one paragraph.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    body: list[int] = []
    for prompt in prompts:
        text = prompt["prompt_text"]
        inner = text.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
        body += tok.encode(inner + "\n\n")
        if len(head) + len(body) + len(tail) >= target:
            break
    body = body[: max(0, target - len(head) - len(tail))]
    return head + body + tail


def run_parallel(url: str, payloads: list[dict[str, Any]], timeout: float) -> tuple[list[dict[str, Any]], float]:
    results: list[Any] = [None] * len(payloads)

    def one(i: int) -> None:
        body, wall = complete(url, payloads[i], timeout)
        results[i] = {"body": body, "wall": wall}

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(payloads))]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started
    if any(r is None for r in results):
        raise SamplerError("a parallel probe request failed")
    return results, wall


def summarize(body: dict[str, Any], wall: float) -> dict[str, Any]:
    t = body.get("timings") or {}
    return {"prompt_n": t.get("prompt_n"), "prompt_per_second": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"), "predicted_per_second": t.get("predicted_per_second"),
            "predicted_ms": t.get("predicted_ms"), "prompt_ms": t.get("prompt_ms"), "wall_s": round(wall, 2),
            "stop_type": body.get("stop_type")}


def probe(args: argparse.Namespace) -> dict[str, Any]:
    tok = ParentTokenizer(Path(args.parent_dir))
    prompts = load_prompts(Path(args.prompts), Path(args.prompts_manifest) if args.prompts_manifest else None)["prompts"]
    sampling = parent_sampling(Path(args.parent_dir))
    url = args.server
    props = server_props(url)
    sampler = CardSampler(args.gpu_index)
    sampler.start()
    cells = {}
    long_ids = filler_ids(tok, prompts, int(args.context))
    decode_n = int(args.decode_tokens)
    # cell 1: one stream, prompt of `context` tokens -> prefill tok/s and decode tok/s @context
    body, wall = complete(url, completion_payload(long_ids, decode_n, 1, sampling, ignore_eos=True), timeout=1800)
    cells["single_stream_at_context"] = dict(summarize(body, wall), prompt_ids=len(long_ids))
    # cell 2: `slots` parallel streams, each at `context` tokens
    payloads = [completion_payload(long_ids, decode_n, 100 + i, sampling, ignore_eos=True)
                for i in range(int(args.slots))]
    results, wall = run_parallel(url, payloads, 3600)
    per = [summarize(r["body"], r["wall"]) for r in results]
    decode_ms = max(p["predicted_ms"] or 0 for p in per)
    cells["parallel_at_context"] = {
        "streams": len(per), "per_stream": per,
        "aggregate_decode_tok_s": round(sum(p["predicted_n"] or 0 for p in per) / (decode_ms / 1000.0), 3) if decode_ms else None,
        "aggregate_wall_tok_s": round(sum(p["predicted_n"] or 0 for p in per) / wall, 3), "wall_s": round(wall, 2)}
    # cell 3: `slots` parallel streams on real self-gen prompts (the run's shape)
    real = [encode_prompt(tok, p) for p in prompts[: int(args.slots)]]
    payloads = [completion_payload(ids, decode_n, 200 + i, sampling, ignore_eos=True) for i, ids in enumerate(real)]
    results, wall = run_parallel(url, payloads, 3600)
    per = [summarize(r["body"], r["wall"]) for r in results]
    decode_ms = max(p["predicted_ms"] or 0 for p in per)
    cells["parallel_real_prompts"] = {
        "streams": len(per), "prompt_ids": [len(r) for r in real], "per_stream": per,
        "aggregate_decode_tok_s": round(sum(p["predicted_n"] or 0 for p in per) / (decode_ms / 1000.0), 3) if decode_ms else None,
        "aggregate_wall_tok_s": round(sum(p["predicted_n"] or 0 for p in per) / wall, 3), "wall_s": round(wall, 2)}
    sampler.halt.set()
    sampler.join(timeout=5)
    receipt = seal({
        "schema": RECEIPT_SCHEMA + "/p5",
        "probe": "P5 llama-server Q4_K_M of the parent on a P40, -np slots (DISTILL_DESIGN S0)",
        "tool": tool_identity(),
        "server": url,
        "props": props,
        "gpu_index": args.gpu_index,
        "card_peak_mib": sampler.peak,
        "context": int(args.context),
        "decode_tokens": decode_n,
        "sampling": sampling,
        "ignore_eos": True,
        "cells": cells,
        "proxy_reference": "P40 decode 11.84 tok/s @4k, prefill 245.5 tok/s (Qwen3.6-27B Q4_K_M, EXPORT_LANE_SPEC 3.3)",
        "measured_utc": utc_now(),
    }, "receipt_sha256")
    return receipt


def sanity(args: argparse.Namespace) -> dict[str, Any]:
    tok = ParentTokenizer(Path(args.parent_dir))
    prompts = load_prompts(Path(args.prompts), Path(args.prompts_manifest) if args.prompts_manifest else None)["prompts"]
    sampling = parent_sampling(Path(args.parent_dir))
    chosen, seen = [], set()
    for index, prompt in enumerate(prompts):
        if prompt["route"] not in seen:
            seen.add(prompt["route"])
            chosen.append((index, prompt))
        if len(chosen) >= int(args.count):
            break
    urls = list(args.server)
    rows = []
    for n, (index, prompt) in enumerate(chosen):
        url = urls[n % len(urls)]
        ids = encode_prompt(tok, prompt)
        server_ids = server_tokenize(url, prompt["prompt_text"])
        seed = prompt_seed(args.base_seed, prompt["rid"])
        body, wall = complete(url, completion_payload(ids, int(args.max_new), seed, sampling), timeout=1800)
        record = build_record(index, prompt, ids, seed, body, wall, url, 1, tok)
        first = next((line for line in record["text"].splitlines() if line.strip()), "")
        print(f"--- [{prompt['route']}] {prompt['rid']} ({url}) stop={record['stop_type']} "
              f"tokens={record['response_tokens']} eog={record['eog_terminated']}", flush=True)
        print("    prompt: " + prompt["prompt_text"].split("<|im_start|>user\n", 1)[-1][:160].replace("\n", " "), flush=True)
        print("    first line: " + first[:300], flush=True)
        rows.append({"index": index, "rid": prompt["rid"], "route": prompt["route"], "server": url,
                     "tokenizer_parity": server_ids == ids, "server_prompt_ids": len(server_ids), "hf_prompt_ids": len(ids),
                     "first_line": first, "record": record, "anomalies": record_anomalies(record)})
    receipt = seal({
        "schema": RECEIPT_SCHEMA + "/sanity",
        "tool": tool_identity(),
        "servers": {u: server_props(u) for u in urls},
        "sampling": sampling,
        "max_new": int(args.max_new),
        "tokenizer_parity": f"{sum(r['tokenizer_parity'] for r in rows)}/{len(rows)}",
        "decode_parity": f"{sum(r['record']['text_matches_server'] for r in rows)}/{len(rows)}",
        "rows": rows,
        "note": "read by a person; the receipt keeps every sample verbatim",
        "measured_utc": utc_now(),
    }, "receipt_sha256")
    return receipt


#: Non-EOG controls for eog-probe: <think>, <tool_call> (added tokens beside the FIM block).
EOG_PROBE_CONTROLS = (248_068, 248_058)


def eog_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Which ids does the LIVE server end generation on?  Each candidate is forced with a
    +100 logit bias for up to 3 tokens: an EOG id stops at once with stop_type "eos", any
    other id repeats to the limit.  The set must equal EOG_IDS + SERVER_ONLY_EOG_IDS, the
    table `end_labels` relies on; anything else is a functional failure (exit 1)."""
    tok = ParentTokenizer(Path(args.parent_dir))
    for token_id, piece in SERVER_ONLY_EOG_PIECES.items():
        if tok.piece(token_id) != piece:
            raise SamplerError(f"tokenizer id {token_id} is {tok.piece(token_id)!r}, expected {piece!r}")
    sampling = parent_sampling(Path(args.parent_dir))
    prompt_ids = tok.encode("<|im_start|>user\nSay one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    candidates = list(EOG_IDS) + list(SERVER_ONLY_EOG_IDS) + list(EOG_PROBE_CONTROLS)
    out = {}
    for url in args.server:
        rows = []
        for n, token_id in enumerate(candidates):
            payload = dict(completion_payload(prompt_ids, 3, 1000 + n, sampling), logit_bias=[[int(token_id), 100.0]])
            body, wall = complete(url, payload, timeout=float(args.request_timeout))
            ids = [int(t) for t in body.get("tokens") or []]
            rows.append({"id": token_id, "piece": tok.piece(token_id), "tokens": ids, "stop_type": body.get("stop_type"),
                         "content": body.get("content"), "forced": bool(ids) and ids[0] == token_id,
                         "ends_generation": ids == [token_id] and body.get("stop_type") == "eos",
                         "wall_s": round(wall, 2)})
            log(f"eog-probe {url} id {token_id} {tok.piece(token_id)!r}: tokens {ids} stop {body.get('stop_type')} "
                f"content {body.get('content')!r}")
        server_eog = sorted(r["id"] for r in rows if r["ends_generation"])
        out[url] = {"props": server_props(url), "rows": rows, "server_eog_ids": server_eog,
                    "all_forced": all(r["forced"] for r in rows)}
    expected = sorted(EOG_IDS + SERVER_ONLY_EOG_IDS)
    ok = all(v["all_forced"] and v["server_eog_ids"] == expected for v in out.values())
    return seal({
        "schema": RECEIPT_SCHEMA + "/eog-probe",
        "tool": tool_identity(),
        "expected_server_eog_ids": expected,
        "hf_eog_ids": list(EOG_IDS),
        "server_only_eog_ids": list(SERVER_ONLY_EOG_IDS),
        "controls": list(EOG_PROBE_CONTROLS),
        "servers": out,
        "ok": ok,
        "measured_utc": utc_now(),
    }, "receipt_sha256")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    def common_parent(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))
        sp.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
        sp.add_argument("--prompts-manifest", default=str(DEFAULT_PROMPTS_MANIFEST))
        sp.add_argument("--base-seed", type=int, default=20261013)

    sp = sub.add_parser("tensor-map")
    sp.add_argument("--dryrun-log", required=True)
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("export-check")
    sp.add_argument("--export-receipt", required=True)
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))
    sp.add_argument("--export-tools", default=str(DEFAULT_EXPORT_TOOLS))
    sp.add_argument("--quant", default="Q4_K_M")
    sp.add_argument("--delete-full", action="store_true", help="rm the full-precision GGUF (declared scratch only)")
    sp.add_argument("--scratch-root", action="append", default=None)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("serve")
    sp.add_argument("--gguf", required=True)
    sp.add_argument("--gpu-index", type=int, required=True)
    sp.add_argument("--allowed-gpus", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_ALLOWED_GPUS))
    sp.add_argument("--port", type=int, required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--slots", type=int, default=4)
    sp.add_argument("--ctx-per-slot", type=int, default=6144)
    sp.add_argument("--threads", type=int, default=8)
    sp.add_argument("--flash-attn", default="auto", choices=("on", "off", "auto"))
    sp.add_argument("--kv-type", default="f16")
    sp.add_argument("--nice", type=int, default=15)
    sp.add_argument("--health-timeout", type=float, default=900)
    sp.add_argument("--llamacpp-root", default=str(DEFAULT_LLAMACPP_ROOT))
    sp.add_argument("--export-tools", default=str(DEFAULT_EXPORT_TOOLS))
    sp.add_argument("--log-dir", default=str(OC / "distill" / "selfgen" / "logs"))
    sp.add_argument("--receipt-dir", default=str(OC / "distill" / "selfgen" / "receipts"))
    sp.add_argument("--allocator-dir", default=str(DEFAULT_ALLOCATOR_DIR))
    sp.add_argument("--purpose", default="self-gen sampler: parent Q4_K_M llama-server (DISTILL_DESIGN 3, U7)")
    sp.add_argument("--extra", nargs=argparse.REMAINDER)

    sp = sub.add_parser("stop")
    sp.add_argument("--launch-receipt", required=True)
    sp.add_argument("--allocator-dir", default=str(DEFAULT_ALLOCATOR_DIR))
    sp.add_argument("--outcome", default=None)

    sp = sub.add_parser("probe")
    common_parent(sp)
    sp.add_argument("--server", required=True)
    sp.add_argument("--gpu-index", type=int, default=None)
    sp.add_argument("--slots", type=int, default=4)
    sp.add_argument("--context", type=int, default=4096)
    sp.add_argument("--decode-tokens", type=int, default=256)
    sp.add_argument("--tag", required=True)
    sp.add_argument("--out-dir", default=str(DEFAULT_TPS_DIR))

    sp = sub.add_parser("sanity")
    common_parent(sp)
    sp.add_argument("--server", action="append", required=True)
    sp.add_argument("--count", type=int, default=5)
    sp.add_argument("--max-new", type=int, default=512)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("eog-probe")
    common_parent(sp)
    sp.add_argument("--server", action="append", required=True)
    sp.add_argument("--request-timeout", type=float, default=1800)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("run")
    common_parent(sp)
    sp.add_argument("--server", action="append", default=[])
    sp.add_argument("--servers-file", default=None, help="extra server URLs, one per line, re-read every progress tick")
    sp.add_argument("--slots-per-server", type=int, default=4)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--export-check-receipt", default=None)
    sp.add_argument("--gguf", default=None, help="fixture only; the real run locks the export-check receipt")
    sp.add_argument("--gguf-sha256", default=None)
    sp.add_argument("--token-target", type=int, default=1_000_000)
    sp.add_argument("--rows-per-shard", type=int, default=64)
    sp.add_argument("--request-timeout", type=float, default=3600)
    sp.add_argument("--retries", type=int, default=4)
    sp.add_argument("--retry-backoff", type=float, default=15)
    sp.add_argument("--max-consecutive-errors", type=int, default=8)
    sp.add_argument("--progress-every", type=float, default=60)
    sp.add_argument("--rate-window-s", type=float, default=1800)
    sp.add_argument("--allow-unpinned-tokenizer", action="store_true", help="fixtures only")
    sp.add_argument("--stop-after", type=int, default=0, help="test hook: stop cleanly after N journal records")

    sp = sub.add_parser("status")
    sp.add_argument("--out-dir", required=True)

    sp = sub.add_parser("verify")
    sp.add_argument("--out-dir", required=True)

    sp = sub.add_parser("export-rescore")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--dest", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        if args.command == "tensor-map":
            out = seal(dict(tensor_map(Path(args.dryrun_log), Path(args.parent_dir)), tool=tool_identity(),
                            schema=RECEIPT_SCHEMA + "/tensor-map"), "receipt_sha256")
            write_json_atomic(Path(args.out), out)
            log(f"tensor map: {out['hf_tensor_count']} HF tensors {out['hf_tensor_classes']} -> {out['gguf_tensor_count']} GGUF "
                f"tensors, {out['gguf_block_count']} blocks, out types {out['gguf_out_types']}; receipt {args.out}")
            return 0
        if args.command == "export-check":
            if args.scratch_root is None:
                args.scratch_root = [str(p) for p in DECLARED_SCRATCH]
            out = export_check(args)
            write_json_atomic(Path(args.out), out)
            log(f"export-check: Q4 {out['q4']['sha256'][:16]} {out['q4']['bytes'] / 1e9:.2f} GB; parent unchanged "
                f"{out['parent_unchanged']} ({out['parent_digest_after']['directory_sha256'][:16]}); full GGUF {out['full_gguf']}")
            return 0
        if args.command == "serve":
            out = serve(args)
            log(f"serve {args.name}: pid {out['pid']} healthy {out['healthy']} in {out['health_wait_s']} s; "
                f"props {out['props']}; log analysis {out['log_analysis']}")
            return 0
        if args.command == "stop":
            out = stop_server(args)
            log(f"stop: {out}")
            return 0
        if args.command == "probe":
            out = probe(args)
            dest = Path(args.out_dir) / f"p5-selfgen-{args.tag}.json"
            write_json_atomic(dest, out)
            c = out["cells"]
            log(f"P5 {args.tag}: single @{out['context']}: prefill {c['single_stream_at_context']['prompt_per_second']} tok/s, "
                f"decode {c['single_stream_at_context']['predicted_per_second']} tok/s; "
                f"{c['parallel_at_context']['streams']} parallel @{out['context']}: aggregate decode "
                f"{c['parallel_at_context']['aggregate_decode_tok_s']} tok/s; real prompts aggregate decode "
                f"{c['parallel_real_prompts']['aggregate_decode_tok_s']} tok/s; card peak {out['card_peak_mib']} MiB; receipt {dest}")
            return 0
        if args.command == "sanity":
            out = sanity(args)
            write_json_atomic(Path(args.out), out)
            log(f"sanity: tokenizer parity {out['tokenizer_parity']}, decode parity {out['decode_parity']}; receipt {args.out}")
            return 0
        if args.command == "eog-probe":
            out = eog_probe(args)
            write_json_atomic(Path(args.out), out)
            log(f"eog-probe: server EOG ids {[v['server_eog_ids'] for v in out['servers'].values()]} vs expected "
                f"{out['expected_server_eog_ids']}; ok {out['ok']}; receipt {args.out}")
            return 0 if out["ok"] else 1
        if args.command == "run":
            return Run(args).execute()
        if args.command == "status":
            return status(Path(args.out_dir))
        if args.command == "verify":
            return verify_run(Path(args.out_dir))
        if args.command == "export-rescore":
            out = export_rescore(Path(args.out_dir), Path(args.dest))
            log(f"export-rescore: {out['rows']} rows x {out['row_length']}, {out['stored_positions']:,} stored positions -> "
                f"{out['corpus']['path']} ({out['corpus']['sha256'][:16]})")
            return 0
    except SamplerError as exc:
        log(f"REFUSED: {exc}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
