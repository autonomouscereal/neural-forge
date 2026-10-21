"""adapters.gguf -- a GGUF served by an ALREADY RUNNING llama-server (the runner never starts one: the P40s
it would need belong to the Flash Next scorer unless Cyril says otherwise).

Reuses toolcall_selfgen exactly as `bank-run` did for the 2026-10-14 parent Q4_K_M bank run: Renderer
(pinned Qwen tokenizer.json + chat template, tools=, thinking off) -> prompt ids -> Client.generate(ids,
role, seed) with SAMPLING["bank"] (greedy) and n_predict set to the run's cap.  The server's /props (model path, slots, build) goes into the receipt.
UNVERIFIED end to end on 2026-10-21 (no llama-server was running and none was started).
"""
from __future__ import annotations

from typing import Any

from . import common as A


class Session:
    engine = "gguf"

    def __init__(self, spec: dict):
        url = spec["device"].get("server_url")
        if not url:
            raise RuntimeError("engine gguf needs --server-url of a running llama-server")
        self.sg = A.load_tool("toolcall_selfgen", A.tool_path(spec, "toolcall_selfgen"))
        tok_dir = spec.get("tokenizer_dir")
        if not tok_dir:
            raise RuntimeError("engine gguf needs --tokenizer-dir (the Qwen parent directory the Renderer pins)")
        self.renderer = self.sg.Renderer(tok_dir, allow_unpinned=bool(spec["gen"].get("allow_unpinned_tokenizer")))
        self.client = self.sg.Client(url, self.renderer)
        self.url = url
        self.seed = int(spec["gen"].get("seed", 1234))
        # the bank-run sampling (greedy: temperature 0, top_k 1) with this run's cap as n_predict
        self.sg.SAMPLING["omni_lab"] = dict(self.sg.SAMPLING["bank"], n_predict=int(spec["gen"]["cap"]))
        try:
            self.props = self.sg.server_props(url)
        except Exception as exc:  # noqa: BLE001 -- recorded; generation will fail loudly if the server is gone
            self.props = {"error": f"{type(exc).__name__}: {exc}"}

    def generate(self, case: dict, gen: dict) -> dict:
        missing = A.non_text_inputs(case)
        if missing:
            return A.unsupported_record(case, f"llama-server path feeds text only; case needs {missing}")
        messages = A.messages_of(case)
        if A.is_raw_text(case):
            ids = self.renderer.encode(case["prompt"]["raw_text"])
        else:
            ids = self.renderer.encode(self.renderer.render(messages, case.get("tools"), True))
        out = self.client.generate(ids, "omni_lab", self.seed)
        term = "complete" if out["end"] in ("eos", "server_only_eog") else "censored-at-cap"
        return {"id": case["case_id"], "category": case.get("category") or "uncategorised", "prompt_messages": messages,
                "prompt_tokens": len(ids), "response_verbatim": out["text"], "generated_tokens": len(out["ids"]),
                "generated_token_ids": out["ids"], "loop_detected": None, "termination": term,
                "runtime_error": None, "wall_time_seconds": out["wall_s"],
                "server": {"end": out["end"], "stop_type": out.get("stop_type"), "url": out["server"]}}

    def report(self) -> dict:
        return {"load": {"server_props": self.props}, "kernels": {"engine": "llama.cpp", "build": self.props.get("build_info")},
                "kernel_path": "llama.cpp-cuda", "placement": {"server_url": self.url, "devices": []},
                "tokenizer": self.renderer.report(), "thinking": {"requested": "off", "took_effect": True,
                                                                  "note": "Renderer always renders enable_thinking=False"},
                "sampling": dict(self.sg.SAMPLING["omni_lab"]), "seed": self.seed}
