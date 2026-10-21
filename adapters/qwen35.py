"""adapters.qwen35 -- a qwen3_5-layout TEXT student (model.language_model.* keys, no vision tower), e.g.
the full-KD run-1 export.  The loader is student_test.load_student, unchanged: stock Qwen3_5ForCausalLM on
meta, the one rename model.language_model. -> model., load_state_dict(strict=True, assign=True), accelerate
dispatch_model.  Only its device plan is generalised from student_test's fixed 3 cards:

    1 card   everything on cuda:0
    2 cards  cuda:0 embed + full-attention layers + norm + head (SDPA); cuda:1 every linear-attention layer
    3 cards  student_test.device_plan itself (cuda:0 as above; linear-attention split over cuda:1/cuda:2)
    N cards  cuda:0 as above; linear-attention layers split in contiguous runs over cuda:1..N-1

Put the sm86 cards (3090s) after the first card so fla's Triton GDN kernels run where the student was
trained with them (receipt-004 kernels); the runner records which card held which layers.
The S8 ledger-band logit mask (ids >= 248,320 to -inf) is on unless the spec turns it off.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import common as A


def make_plan(n_cards: int, original):
    """A device_plan(layer_types) for n_cards; n_cards == 3 returns student_test's own function."""
    if n_cards == 3:
        return original

    def plan_fn(layer_types: list[str]) -> dict[str, int]:
        plan: dict[str, int] = {"model.embed_tokens": 0}
        linear = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
        workers = max(1, n_cards - 1)
        per = -(-len(linear) // workers) if linear else 1
        for index, kind in enumerate(layer_types):
            if n_cards == 1 or kind == "full_attention":
                plan[f"model.layers.{index}"] = 0
            else:
                plan[f"model.layers.{index}"] = 1 + linear.index(index) // per
        plan.update({"model.norm": 0, "model.rotary_emb": 0, "lm_head": 0})
        return plan

    return plan_fn


class Session:
    engine = "qwen35-student"

    def __init__(self, spec: dict):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        dev = spec["device"]
        if dev["kind"] != "cuda" or not dev.get("cards"):
            raise RuntimeError("engine qwen35-student needs CUDA cards (student_test.load_student is CUDA-only)")
        cal = A.tool_path(spec, "calibration_dir")
        self.oc_calib = A.load_tool("oc_calib", f"{cal}/oc_calib.py", add_dir_to_path=True)
        self.st = A.load_tool("student_test", A.tool_path(spec, "student_test"))
        n = torch.cuda.device_count()
        if n != len(dev["cards"]):
            raise RuntimeError(f"{n} CUDA devices visible, spec names {len(dev['cards'])}")
        self.st.device_plan = make_plan(n, self.st.device_plan)
        self.model, self.cfg, self.load_report = self.st.load_student(Path(spec["model_path"]), torch)
        tok_dir = spec.get("tokenizer_dir")
        if not tok_dir:
            raise RuntimeError("engine qwen35-student needs a tokenizer directory (the student export carries none)")
        self.tok_dir = tok_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        names = list(spec["gen"]["extra_eos_tokens"])
        self.stops = [int(self.tokenizer.convert_tokens_to_ids(t)) for t in names]
        self.eos_id = self.stops[0] if self.stops else self.tokenizer.eos_token_id
        band = self.st.LEDGER_BAND_BEGIN if spec["gen"].get("ledger_band_mask", True) else A.NO_BAND
        self.band = band
        self.wrapper = self.st.MaskedGreedyModel(self.model, band)
        self.thinking = self.oc_calib.thinking_report(self.tokenizer, [{"role": "user", "content": "Hello."}],
                                                      spec["gen"]["thinking"])
        self.kernel_report = self.oc_calib.kernel_path_report(self.model)

    def generate(self, case: dict, gen: dict) -> dict:
        missing = A.non_text_inputs(case)
        if missing:
            return A.unsupported_record(case, f"the qwen3_5 student is a text-only export (no vision tower); "
                                              f"case needs {missing}")
        return A.greedy_case(oc_calib=self.oc_calib, student_test=self.st, wrapper=self.wrapper,
                             tokenizer=self.tokenizer, case=case, gen=gen, device="cuda:0", eos_id=self.eos_id,
                             stops=self.stops, label="STUDENT")

    def report(self) -> dict:
        types = list(getattr(self.cfg, "layer_types", []) or [])
        plan = (self.load_report.get("placement") or {}).get("plan") or {}
        devices = []
        for card in (self.load_report.get("placement") or {}).get("devices") or []:
            i = card["cuda"]
            layers = sorted(int(k.rsplit(".", 1)[1]) for k, v in plan.items() if k.startswith("model.layers.") and v == i)
            kinds = {types[j] for j in layers if j < len(types)}
            role = ("full-attn" if kinds == {"full_attention"} else "linear-attn" if kinds == {"linear_attention"}
                    else "compute")
            kpath = {"full-attn": f"sdpa-{card.get('capability')}",
                     "linear-attn": f"fla-triton-{card.get('capability')}"}.get(role, "mixed")
            devices.append({"cuda": i, "role": role, "kernel_path": kpath, "layers": layers,
                            "other_modules": sorted(k for k, v in plan.items() if v == i and not k.startswith("model.layers."))})
        load = {k: v for k, v in self.load_report.items() if k not in ("kernels", "placement")}
        return {"load": load, "kernels": {**(self.load_report.get("kernels") or {}), "oc_calib": self.kernel_report},
                "kernel_path": self.kernel_report.get("kernel_path"),
                "placement": {"plan": plan, "devices": devices},
                "tokenizer": {"dir": self.tok_dir, "tokenizer_json_sha256": A.file_sha256(Path(self.tok_dir) / "tokenizer.json"),
                              "chat_template_sha256": A.file_sha256(Path(self.tok_dir) / "chat_template.jinja")},
                "thinking": self.thinking, "eos_id": self.eos_id, "stops": self.stops,
                "ledger_band_mask": self.band if self.band != A.NO_BAND else None}
