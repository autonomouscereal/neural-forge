"""Shared pieces of the fold research lane (L-FOLD-RESEARCH, Claude): hashing, the eval screen, the tool-call
transport parser, the loop detector, the horizon controller, paired tests, model loading, generation and the two
distillation losses.  Written 2026-10-14/15 for pilot X3 (FOLDING_PROGRAM_v2 + its adversarial critique).

Grading rules this lane follows (owner, memory `folding-behavior-first-grading.md`): token match is never a gate,
a grade or an optimisation target.  Free-running retention and artifact counts (empty / cap / loop) are reported as
DIAGNOSTICS.  The primary block is content: Codex's independent oracles (arithmetic, pure-function code, strict
JSON, BFCL call proposals), graded artifact-tolerant, plus the 15-case bank v1 text for a human reader.

Torch / transformers are imported lazily so the pure logic (screen, parser, loop detector, controller, tests)
runs on any CPU Python.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TOOL_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from chimera_recovery import evaluation as cx_eval  # noqa: E402  (pinned, read-only)
from chimera_recovery import folding_retention as cx_ret  # noqa: E402

VOCAB = 248_320


# ----------------------------------------------------------------------------------------------- identity / hashing

def canonical(value: Any) -> bytes:
    """Byte-identical to Codex `folding_retention.canonical`, so digests agree across lanes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


IDENTITY_FILES = re.compile(r"(\.safetensors|config\.json|tokenizer\.json|tokenizer_config\.json|chat_template\.jinja"
                            r"|generation_config\.json|model\.safetensors\.index\.json)$")


def dir_digest(model_dir: str | Path) -> dict[str, Any]:
    """sha256 over sorted (relative path, file sha256) of the weight / config / tokenizer files of a checkpoint."""
    root = Path(model_dir)
    pairs = sorted((p.relative_to(root).as_posix(), file_sha256(p)) for p in root.rglob("*")
                   if p.is_file() and IDENTITY_FILES.search(p.name) and ".cache" not in p.parts)
    if not pairs:
        raise ValueError(f"no checkpoint files under {root}")
    return {"sha256": digest(pairs), "files": len(pairs)}


def checkpoint_identity(model_dir: str | Path, overlay: str | Path | None) -> dict[str, Any]:
    base = dir_digest(model_dir)
    if overlay is None:
        return {"model_dir": str(model_dir), "dir_sha256": base["sha256"], "overlay": None,
                "checkpoint_sha256": base["sha256"]}
    osha = file_sha256(overlay)
    return {"model_dir": str(model_dir), "dir_sha256": base["sha256"], "overlay": str(overlay),
            "overlay_sha256": osha, "checkpoint_sha256": digest({"dir": base["sha256"], "overlay": osha})}


def verify_vendor_pins() -> dict[str, str]:
    pins = json.loads((HERE / "VENDOR_PINS.json").read_text(encoding="utf-8"))
    seen = {}
    for rel, want in pins["files"].items():
        have = file_sha256(VENDOR / rel)
        if have != want:
            raise SystemExit(f"vendored {rel} drifted: {have} != pinned {want}")
        seen[rel] = have
    return seen


def own_shas() -> dict[str, str]:
    return {p.name: file_sha256(p) for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh"))}


def peak_rss_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


def mem_available_gib(meminfo: str = "/proc/meminfo") -> float:
    with open(meminfo, encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    raise RuntimeError("MemAvailable missing")


def write_json(path: str | Path, body: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False).encode() + b"\n")
    os.replace(tmp, path)
    return file_sha256(path)


def salted_rank(salt: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}|{key}".encode()).digest()[:8], "big")


# ------------------------------------------------------------------------------------------------- eval screen

#: the CJK-aware word rule of distill_corpus_build 1.0.1 (WORD_RE, line 175), so this screen and the corpus screen
#: split text the same way
CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
WORD_RE = re.compile(f"[{CJK}]|[^\\W_{CJK}]+")
NGRAM = 13
SHORT_MIN_WORDS = 5


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


class Screen:
    """Eval-surface screen for training text.

    A training text HITS an eval item when (a) its normalised word sequence equals the item's, (b) more than half of
    the item's 13-grams occur in it (Codex text_eval_seed rule: >50% candidate 13-gram overlap), or (c) the item is
    short (5-12 words) and appears in it as a contiguous word run.  Lexical only: it does not establish semantic
    novelty (Codex's own caveat), which is why retention prompts also come from a carve split that the corpus
    builder never trains on.
    """

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.gram_index: dict[tuple, set[str]] = defaultdict(set)
        self.exact: dict[tuple, set[str]] = defaultdict(set)
        self.short: dict[str, str] = {}
        self.sources: dict[str, int] = defaultdict(int)

    def add(self, source: str, ident: str, text: str) -> None:
        toks = tuple(words(text))
        if not toks:
            return
        key = f"{source}::{ident}"
        grams = {toks[i:i + NGRAM] for i in range(len(toks) - NGRAM + 1)}
        self.items[key] = {"source": source, "grams": len(grams), "words": len(toks)}
        self.exact[toks].add(key)
        for gram in grams:
            self.gram_index[gram].add(key)
        if not grams and len(toks) >= SHORT_MIN_WORDS:
            self.short[key] = " " + " ".join(toks) + " "
        self.sources[source] += 1

    def hits(self, text: str) -> list[dict[str, Any]]:
        toks = tuple(words(text))
        found = []
        for key in self.exact.get(toks, ()):
            found.append({"item": key, "rule": "exact"})
        counts: dict[str, int] = defaultdict(int)
        for gram in {toks[i:i + NGRAM] for i in range(len(toks) - NGRAM + 1)}:
            for key in self.gram_index.get(gram, ()):
                counts[key] += 1
        for key, count in counts.items():
            if count / self.items[key]["grams"] > 0.5:
                found.append({"item": key, "rule": "13gram>50%", "fraction": round(count / self.items[key]["grams"], 4)})
        if self.short:
            joined = " " + " ".join(toks) + " "
            for key, run in self.short.items():
                if run in joined:
                    found.append({"item": key, "rule": "short-contained"})
        return found

    def receipt(self) -> dict[str, Any]:
        return {"items": len(self.items), "by_source": dict(self.sources), "ngram": NGRAM,
                "rules": ["exact normalised", "13-gram > 50% of the eval item's grams", "5-12 word item contained"]}


# ------------------------------------------------------------------------------------- tool-call transport parser

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.S)
XML_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
XML_FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text)
    if "</think>" in text:          # an unopened close tag (the template opened it in the prompt)
        text = text.split("</think>")[-1]
    return text.strip()


def _loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _normalise_call(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    fn = item.get("function") if isinstance(item.get("function"), dict) else {}
    name = item.get("name") or fn.get("name")
    if name is None and len(item) == 1:          # BFCL-native {"func": {"arg": v}}
        (only_key, only_val), = item.items()
        if isinstance(only_val, dict):
            return {"name": only_key, "arguments": only_val}
        return None
    if not isinstance(name, str):
        return None
    for key in ("arguments", "parameters", "args"):
        if key in item:
            return {"name": name, "arguments": _loads_maybe(item[key])}
    if "arguments" in fn:
        return {"name": name, "arguments": _loads_maybe(fn["arguments"])}
    return {"name": name, "arguments": {}}


def _calls_from_json(value: Any) -> list[dict] | None:
    if isinstance(value, dict):
        for key in ("tool_calls", "calls", "function_calls", "functions"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            call = _normalise_call(value)
            return [call] if call else None
    if isinstance(value, list) and value:
        calls = [_normalise_call(v) for v in value]
        return calls if all(calls) else None
    return None


def _first_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(text[start:])
                return value
            except ValueError:
                continue
    raise ValueError("no JSON value")


def parse_tool_calls(text: str) -> dict[str, Any]:
    """Raw model text -> proposed calls.  Documented transport parser (EVALUATION_PROTOCOL: 'any transport parser
    must retain raw output and document exact parsing').  Order of attempts, first success wins:
      1. Qwen XML  <tool_call><function=N><parameter=K>V</parameter></function></tool_call>  (V parsed as JSON if it
         parses, else kept as a string) or <tool_call>{json}</tool_call>
      2. the whole answer (think block stripped, one ```json fence unwrapped) as JSON
      3. the first JSON value found in the answer
    JSON shapes accepted: a call {name, arguments|parameters|args} or {function:{name, arguments}}, a BFCL-native
    {func: {arg: v}}, a list of those, or {tool_calls|calls|function_calls|functions: [...]}.  String arguments are
    JSON-decoded.  Nothing is repaired: an unparsable answer yields no calls."""
    body = strip_think(text)
    xml = XML_CALL_RE.findall(body)
    if xml:
        calls = []
        for chunk in xml:
            fns = XML_FN_RE.findall(chunk)
            if fns:
                for name, params in fns:
                    calls.append({"name": name.strip(), "arguments": {k: _loads_maybe(v) for k, v in XML_PARAM_RE.findall(params)}})
            else:
                try:
                    got = _calls_from_json(json.loads(chunk.strip()))
                except ValueError:
                    got = None
                if not got:
                    return {"calls": None, "route": "xml_unparsable"}
                calls.extend(got)
        return {"calls": calls, "route": "qwen_xml"}
    fences = FENCE_RE.findall(body)
    candidate = fences[0].strip() if len(fences) == 1 else body
    for route, getter in (("whole_json", lambda: json.loads(candidate)), ("first_json", lambda: _first_json(body))):
        try:
            calls = _calls_from_json(getter())
        except ValueError:
            continue
        if calls:
            return {"calls": calls, "route": route}
    return {"calls": None, "route": "none"}


def tool_messages(text: str) -> tuple[list[dict], dict[str, Any]]:
    """Codex grade_tools input: one assistant message carrying every proposed call as ONE batch."""
    parsed = parse_tool_calls(text)
    if not parsed["calls"]:
        return [{"role": "assistant", "content": strip_think(text)}], parsed
    tool_calls = []
    for i, call in enumerate(parsed["calls"]):
        try:
            arguments = json.dumps(call["arguments"], ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            arguments = "{"            # stays an invalid JSON string -> Codex grader records the error
        tool_calls.append({"id": f"call_{i}", "type": "function", "function": {"name": call["name"], "arguments": arguments}})
    return [{"role": "assistant", "content": "", "tool_calls": tool_calls}], parsed


# ------------------------------------------------------------------------------------------- artifact tolerance

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
FINAL_RE = re.compile(r"(?i)(?:final answer|answer)\s*[:=]\s*")


def tolerant_text(text: str) -> str:
    """Formatting-only normalisation for the artifact-tolerant arithmetic column: drop the think block, markdown
    emphasis, inline LaTeX delimiters, \\boxed{}, a leading 'Answer:' on the last line and a trailing period.  It
    never searches the body for the expected value."""
    body = strip_think(text)
    body = BOXED_RE.sub(r"\1", body)
    body = body.replace("**", "").replace("__", "").replace("$", "").replace("\\(", "").replace("\\)", "")
    lines = [l.strip() for l in body.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    last = FINAL_RE.sub("", lines[-1]).strip().rstrip(".").strip()
    return "\n".join(lines[:-1] + [last])


# ------------------------------------------------------------------------------ loop detector + horizon control

def detect_loop(ids: list[int], *, min_len: int = 64, max_period: int = 10, anchor: int = 32,
                agree: float = 0.9, min_cycles: int = 3, stride: int = 4) -> int | None:
    """ShortOPD-style repetition detector (periods 1-10, 32-token anchor, >= 90% agreement, >= 3 cycles, only after
    64 tokens).  Returns the end position of the first window that confirms a loop, else None.  Checked every
    `stride` tokens, so the reported position may be up to stride-1 tokens late."""
    n = len(ids)
    if n < max(min_len, anchor):
        return None
    for end in range(max(min_len, anchor), n + 1, stride):
        window_start = end - anchor
        for period in range(1, max_period + 1):
            if anchor < min_cycles * period:
                continue
            same = sum(1 for i in range(window_start + period, end) if ids[i] == ids[i - period])
            if same / (anchor - period) >= agree:
                return end
    return None


class HorizonController:
    """Rollout budget for OPD (the X2d controller folded into X3b per the critique): start short, lengthen while
    rollouts are coherent but still capped, shorten when loops rise.  EMA over steps; budgets are multiples of 64."""

    def __init__(self, start: int = 256, low: int = 128, high: int = 1024, decay: float = 0.7) -> None:
        self.budget, self.low, self.high, self.decay = start, low, high, decay
        self.ema_loop = 0.0
        self.ema_cap = 0.0
        self.history: list[dict] = []

    def update(self, rollouts: int, loops: int, caps: int) -> int:
        if rollouts <= 0:
            return self.budget
        d = self.decay
        self.ema_loop = d * self.ema_loop + (1 - d) * loops / rollouts
        self.ema_cap = d * self.ema_cap + (1 - d) * caps / rollouts
        old = self.budget
        if self.ema_loop > 0.25:
            self.budget = max(self.low, int(self.budget * 0.75) // 64 * 64)
        elif self.ema_cap > 0.5 and self.ema_loop < 0.10:
            self.budget = min(self.high, int(math.ceil(self.budget * 1.5 / 64)) * 64)
        self.history.append({"loops": loops, "caps": caps, "n": rollouts, "ema_loop": round(self.ema_loop, 4),
                             "ema_cap": round(self.ema_cap, 4), "budget_before": old, "budget_after": self.budget})
        return self.budget

    def state(self) -> dict:
        return {"budget": self.budget, "ema_loop": self.ema_loop, "ema_cap": self.ema_cap}

    def load(self, state: dict) -> None:
        self.budget, self.ema_loop, self.ema_cap = int(state["budget"]), float(state["ema_loop"]), float(state["ema_cap"])


# ------------------------------------------------------------------------------------------------ paired tests

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (baseline-only passes) and c (candidate-only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ----------------------------------------------------------------------------------------- model / generation

def force_torch_kernels() -> dict[str, bool]:
    """Same switch as qwen-fold/qf_arch.force_torch_kernels: the P40 (sm61) has no Triton, so the qwen3_5 fused GDN /
    conv / norm kernels are disabled in place and the torch fallbacks in the same module run instead."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    was = {name: getattr(mod, name, None) is not None for name in
           ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
            "causal_conv1d_fn", "causal_conv1d_update")}
    for name in was:
        setattr(mod, name, None)
    mod.is_fast_path_available = False
    return was


def load_tokenizer(model_dir: str | Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    eos = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")})
    if any(e is None or e < 0 for e in eos):
        raise SystemExit("tokenizer lacks <|im_end|> / <|endoftext|>")
    return tok, eos


def chat_ids(tok, messages: list[dict], *, thinking: bool = False) -> list[int]:
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def load_model(model_dir: str | Path, *, device: str, dtype: str = "float32", overlay: str | Path | None = None,
               attn: str = "eager", torch_kernels: bool = True):
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    disabled = force_torch_kernels() if torch_kernels else {}
    config = AutoConfig.from_pretrained(str(model_dir))
    cls = mod.Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else mod.Qwen3_5ForCausalLM
    model = cls.from_pretrained(str(model_dir), dtype=getattr(torch, dtype), attn_implementation=attn)
    model = model.to(device).eval()
    report = {"class": cls.__name__, "dtype": dtype, "attn": attn, "kernels_disabled": disabled}
    if overlay is not None:
        report["overlay"] = apply_overlay(model, overlay)
    return model, report


def mlp_parameters(model) -> dict[str, Any]:
    """The folded component: every gate/up/down projection of the text trunk (never the vision tower)."""
    named = {n: p for n, p in model.named_parameters()
             if re.search(r"layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", n) and "visual" not in n}
    if not named:
        raise SystemExit("no trunk MLP parameters found")
    return named


def apply_overlay(model, overlay: str | Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    state = load_file(str(overlay))
    params = mlp_parameters(model)
    if set(state) != set(params):
        raise SystemExit(f"overlay keys != trunk MLP keys ({len(state)} vs {len(params)})")
    with torch.no_grad():
        for name, tensor in state.items():
            if tuple(tensor.shape) != tuple(params[name].shape):
                raise SystemExit(f"overlay shape mismatch at {name}")
            params[name].copy_(tensor.to(params[name].dtype))
    return {"tensors": len(state), "sha256": file_sha256(overlay)}


def save_overlay(model, path: str | Path) -> str:
    from safetensors.torch import save_file
    params = mlp_parameters(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({n: p.detach().to("cpu").contiguous() for n, p in params.items()}, str(tmp))
    os.replace(tmp, path)
    return file_sha256(path)


def generate(model, prompts: list[list[int]], *, max_new: int, eos_ids: list[int], pad_id: int, batch_size: int,
             do_sample: bool = False, temperature: float = 1.0, seed: int | None = None) -> list[dict]:
    """Left-padded HF generate.  Returns per prompt {generated_ids (through the first EOS-set id, inclusive),
    stop_reason eos|max_tokens}.  Greedy unless do_sample."""
    import torch
    device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[dict | None] = [None] * len(prompts)
    eos = set(eos_ids)
    if seed is not None:
        torch.manual_seed(seed)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            p = prompts[i]
            ids[row, width - len(p):] = torch.tensor(p, dtype=torch.long)
            mask[row, width - len(p):] = 1
        kwargs = dict(max_new_tokens=max_new, eos_token_id=list(eos_ids), pad_token_id=pad_id, use_cache=True,
                      do_sample=do_sample)
        if do_sample:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        else:
            kwargs.update(temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            seq = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), **kwargs)
        for row, i in enumerate(idx):
            gen = [int(t) for t in seq[row, width:].tolist()]
            cut = next((k for k, t in enumerate(gen) if t in eos), None)
            if cut is not None:
                out[i] = {"generated_ids": gen[:cut + 1], "stop_reason": "eos"}
            else:
                out[i] = {"generated_ids": gen[:max_new], "stop_reason": "max_tokens"}
    return out  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------------------- losses

def bucket_forward_kl(student_logits, teacher_logits, k: int = 64):
    """Forward KL(teacher || student) on the teacher's top-k tokens plus one tail bucket (the D2 bucket shape).
    Inputs [N, V] (any float dtype; computed in fp32).  Returns per-position KL [N]."""
    import torch
    t = torch.log_softmax(teacher_logits.float(), dim=-1)
    s = torch.log_softmax(student_logits.float(), dim=-1)
    t_top, idx = t.topk(k, dim=-1)
    s_top = s.gather(-1, idx)
    p_top = t_top.exp()
    p_tail = (1.0 - p_top.sum(-1)).clamp_min(1e-12)
    q_tail = (1.0 - s_top.exp().sum(-1)).clamp_min(1e-12)
    return (p_top * (t_top - s_top)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())


def bucket_jsd(student_logits, teacher_logits, *, k: int = 100, beta: float = 0.5):
    """Generalised JSD_beta(teacher, student) on the teacher's top-k + tail bucket:
    beta*KL(p||m) + (1-beta)*KL(q||m), m = beta*p + (1-beta)*q.  beta=0.5 is the symmetric JSD of Codex's
    `--onpolicy-divergence jsd --jsd-beta .5`.  Returns per-position values [N]."""
    import torch
    t = torch.log_softmax(teacher_logits.float(), dim=-1)
    s = torch.log_softmax(student_logits.float(), dim=-1)
    t_top, idx = t.topk(k, dim=-1)
    s_top = s.gather(-1, idx)
    log_p = torch.cat([t_top, (1.0 - t_top.exp().sum(-1, keepdim=True)).clamp_min(1e-12).log()], dim=-1)
    log_q = torch.cat([s_top, (1.0 - s_top.exp().sum(-1, keepdim=True)).clamp_min(1e-12).log()], dim=-1)
    log_m = torch.logsumexp(torch.stack([log_p + math.log(beta), log_q + math.log(1.0 - beta)]), dim=0)
    kl_pm = (log_p.exp() * (log_p - log_m)).sum(-1)
    kl_qm = (log_q.exp() * (log_q - log_m)).sum(-1)
    return beta * kl_pm + (1.0 - beta) * kl_qm
