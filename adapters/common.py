"""adapters.common -- pieces every torch adapter shares: loading the campaign's own tool files by path,
the prompt shims, the modality gate and the one greedy call.

Nothing here re-implements decoding: `greedy_case` is `oc_calib.generate_greedy` over
`student_test.MaskedGreedyModel` (the student review's per-forward argmax trace + ledger-band mask),
exactly as student_test.py 1.0.0 ran it on 2026-10-21.  The shims are the ones student_test already
uses (`ToolTemplateTokenizer`) plus one for raw continuations (probe prompts), which renders the text
unchanged instead of through a chat template.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

#: no-mask value for MaskedGreedyModel: a band that begins past any vocabulary masks nothing
NO_BAND = 1 << 62
TEXT_MODALITIES = {"text"}

_LOADED: dict[str, Any] = {}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"{utc()} {message}", flush=True)


def file_sha256(path: str | Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def load_tool(name: str, path: str | Path, add_dir_to_path: bool = False) -> Any:
    """Import a campaign tool file by path (once per process) and remember its sha256 for the receipt."""
    path = Path(path)
    if name in _LOADED:
        return _LOADED[name]
    if not path.is_file():
        raise FileNotFoundError(f"tool {name}: {path} does not exist")
    if add_dir_to_path and str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _LOADED[name] = module
    TOOL_SHAS[name] = {"path": str(path), "sha256": file_sha256(path)}
    return module


#: {tool name: {path, sha256}} of every campaign file this process imported; goes into the receipt
TOOL_SHAS: dict[str, dict] = {}


def tool_path(spec: dict, key: str) -> str:
    value = (spec.get("paths") or {}).get(key)
    if not value:
        raise KeyError(f"spec.paths.{key} is not set")
    return value


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def is_raw_text(case: dict) -> bool:
    return isinstance(case.get("prompt"), dict) and "raw_text" in case["prompt"]


def messages_of(case: dict) -> list[dict]:
    p = case.get("prompt")
    if isinstance(p, list):
        return [dict(m) for m in p]
    if isinstance(p, dict) and "raw_text" in p:
        return [{"role": "user", "content": str(p["raw_text"])}]
    if isinstance(p, str):
        return [{"role": "user", "content": p}]
    raise ValueError(f"case {case.get('case_id')}: prompt is neither messages nor raw_text")


def non_text_inputs(case: dict) -> list[str]:
    """The input modalities this case needs beyond text: declared modality_in, media refs, typed content parts."""
    need = set(case.get("modality_in") or ["text"]) - TEXT_MODALITIES
    if case.get("media"):
        for ref in case["media"] if isinstance(case["media"], list) else [case["media"]]:
            if isinstance(ref, dict) and ref.get("kind"):
                need.add(str(ref["kind"]))
    p = case.get("prompt")
    if isinstance(p, list):
        for m in p:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") not in (None, "text"):
                        need.add(str(part.get("type")))
    return sorted(need)


def unsupported_record(case: dict, reason: str, flag: str = "unsupported-modality") -> dict:
    """A result row that says the engine could not take this case; never a silent skip."""
    return {"id": case["case_id"], "category": case.get("category") or "uncategorised",
            "prompt_messages": case.get("prompt"), "prompt_tokens": None, "response_verbatim": "",
            "generated_tokens": 0, "generated_token_ids": [], "loop_detected": False,
            "termination": flag, "runtime_error": None, "wall_time_seconds": 0.0, "unsupported_reason": reason}


class RawTextTokenizer:
    """Renders a raw continuation unchanged where generate_greedy would apply the chat template
    (probe prompts are continuations, not chat turns)."""

    chat_template = "raw-continuation"

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def apply_chat_template(self, messages: Any, **_kwargs: Any) -> str:
        return "".join(str(m.get("content", "")) for m in messages)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


# ---------------------------------------------------------------------------
# the one greedy call
# ---------------------------------------------------------------------------

def greedy_case(*, oc_calib: Any, student_test: Any, wrapper: Any, tokenizer: Any, case: dict, gen: dict,
                device: str, eos_id: int | None, stops: list[int], label: str) -> dict:
    """oc_calib.generate_greedy over the MaskedGreedyModel wrapper for one case; adds the generated ids
    (from the wrapper's argmax trace, checked against the runner's own sha), decode speed and the form box."""
    tok = tokenizer
    if is_raw_text(case):
        tok = RawTextTokenizer(tokenizer)
    elif case.get("tools"):
        tok = student_test.ToolTemplateTokenizer(tokenizer, case["tools"])
    row = {"id": case["case_id"], "category": case.get("category") or "uncategorised", "messages": messages_of(case)}
    wrapper.reset()
    record = oc_calib.generate_greedy(wrapper, tok, [row], cap=int(gen["cap"]), device=device, eos_id=eos_id,
                                      label=label, extra_eos_ids=stops, use_logits_to_keep=True,
                                      thinking="template-default" if is_raw_text(case) else gen["thinking"])[0]
    trace = list(wrapper.trace)
    generated = trace[:-1] if record["termination"] == "complete" else trace
    record["generated_token_ids"] = generated
    record["generated_ids_match_runner_sha"] = (
        oc_calib.canonical_sha256({"ids": generated}) == record["generated_token_ids_sha256"])
    decode_s = sum(wrapper.seconds[1:])
    record["prefill_seconds"] = round(wrapper.seconds[0], 3) if wrapper.seconds else None
    record["decode_tok_per_s"] = round((len(wrapper.seconds) - 1) / decode_s, 3) if decode_s > 0 else None
    record["form"] = oc_calib.classify_form(response=record["response_verbatim"],
                                            payload=oc_calib.user_turns_text(record["prompt_messages"]),
                                            termination=record["termination"], token_ids=generated)
    return record


def torch_env(torch: Any) -> dict:
    """Runtime versions + the cards this process sees (uuid when torch exposes it)."""
    import platform
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": getattr(torch.version, "cuda", None)}
    for mod in ("transformers", "fla", "triton", "accelerate", "safetensors", "tokenizers"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:  # noqa: BLE001 -- an absent optional package is a fact, not an error
            env[mod] = None
    cards = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            uuid = getattr(props, "uuid", None)
            cards.append({"cuda": i, "name": props.name, "capability": "sm%d%d" % (props.major, props.minor),
                          "total_mib": int(props.total_memory // (1 << 20)),
                          "uuid": ("GPU-" + str(uuid)) if uuid is not None else None})
    env["cards"] = cards
    env["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return env


def peak_mib(torch: Any) -> list[int]:
    if not torch.cuda.is_available():
        return []
    return [int(torch.cuda.max_memory_allocated(i) // (1 << 20)) for i in range(torch.cuda.device_count())]
