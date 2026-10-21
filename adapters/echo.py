"""adapters.echo -- MECHANICS BED, not a model.  Returns reference.echo_response when the case carries one,
else the last user turn verbatim.  Exists so the runner -> worker -> graders -> ingest -> compare path can be
tested without torch or a GPU; every run it produces is labelled engine 'echo' and its model row is
whatever the caller named.  Nothing measured on it is a measurement of anything.
"""
from __future__ import annotations

import time

from . import common as A


class Session:
    engine = "echo"

    def __init__(self, spec: dict):
        self.spec = spec

    def generate(self, case: dict, gen: dict) -> dict:
        missing = A.non_text_inputs(case)
        if missing:
            return A.unsupported_record(case, f"echo feeds text only; case needs {missing}")
        started = time.monotonic()
        messages = A.messages_of(case)
        ref = case.get("reference") or {}
        text = ref.get("echo_response") if isinstance(ref, dict) and "echo_response" in ref else \
            next((str(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
        words = text.split()
        cap = int(gen["cap"])
        term = "complete" if len(words) <= cap else "censored-at-cap"
        text = " ".join(words[:cap]) if term != "complete" else text
        return {"id": case["case_id"], "category": case.get("category") or "uncategorised", "prompt_messages": messages,
                "prompt_tokens": sum(len(str(m.get("content", "")).split()) for m in messages),
                "response_verbatim": text, "generated_tokens": len(text.split()),
                "generated_token_ids": list(range(len(text.split()))), "loop_detected": False, "termination": term,
                "runtime_error": None, "wall_time_seconds": round(time.monotonic() - started, 4)}

    def report(self) -> dict:
        return {"load": {"is_mechanics_bed": True}, "kernels": {}, "kernel_path": "none",
                "placement": {"devices": []}, "thinking": {"requested": self.spec["gen"]["thinking"], "took_effect": False},
                "tokenizer": None}
