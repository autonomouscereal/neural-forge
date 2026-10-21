"""adapters.hf -- a stock transformers checkpoint (config.json architectures[0]) on ONE device.

Loader: oc_run_frozen_bank.load_model (stock class read off the checkpoint, trust_remote_code false,
refuses a class mismatch), the same function the HF calibration runner used for the S0/Q35 dossiers.
Stops: oc_run_frozen_bank.stop_token_ids(<|im_end|>, <|endoftext|>).  On CPU the qwen3_5 fused kernels
(Triton / CUDA-constructing FusedRMSNormGated) are switched to the torch fallbacks first with
qf_arch.force_torch_kernels, and the receipt says which were switched.
Multi-card HF placement is not done here: a model that needs more than one card goes through
qwen35-student (qwen3_5 layout) or chimera.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import common as A


class Session:
    engine = "hf"

    def __init__(self, spec: dict):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        cal = A.tool_path(spec, "calibration_dir")
        self.oc_calib = A.load_tool("oc_calib", f"{cal}/oc_calib.py", add_dir_to_path=True)
        ocr = A.load_tool("oc_run_frozen_bank", f"{cal}/oc_run_frozen_bank.py", add_dir_to_path=True)
        self.st = A.load_tool("student_test", A.tool_path(spec, "student_test"))
        dev = spec["device"]
        if dev["kind"] == "cuda" and len(dev.get("cards") or []) != 1:
            raise RuntimeError("engine hf runs on exactly one card (or --cpu); use qwen35-student / chimera for splits")
        self.device = "cpu" if dev["kind"] == "cpu" else "cuda:0"
        self.forced = None
        if self.device == "cpu":
            qf = A.load_tool("qf_arch", A.tool_path(spec, "qf_arch"))
            self.forced = qf.force_torch_kernels()
            A.log(f"cpu: qwen3_5 fused kernels switched to torch fallbacks (were present: {self.forced})")
        if dev.get("threads"):
            torch.set_num_threads(int(dev["threads"]))
        dtype = self.oc_calib.resolve_dtype(spec["dtype"])
        self.model, self.export = ocr.load_model(Path(spec["model_path"]), dtype=dtype, device=self.device,
                                                 requested_class=spec.get("loader_class"))
        tok_dir = spec.get("tokenizer_dir") or spec["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        self.stops, self.stop_names = ocr.stop_token_ids(self.tokenizer, list(spec["gen"]["extra_eos_tokens"]))
        self.eos_id = self.tokenizer.eos_token_id
        self.wrapper = self.st.MaskedGreedyModel(self.model, A.NO_BAND)
        self.thinking = self.oc_calib.thinking_report(self.tokenizer, [{"role": "user", "content": "Hello."}],
                                                      spec["gen"]["thinking"])
        self.tok_dir = tok_dir

    def generate(self, case: dict, gen: dict) -> dict:
        missing = A.non_text_inputs(case)
        if missing:
            return A.unsupported_record(case, f"engine hf feeds text only; case needs {missing}")
        return A.greedy_case(oc_calib=self.oc_calib, student_test=self.st, wrapper=self.wrapper,
                             tokenizer=self.tokenizer, case=case, gen=gen, device=self.device, eos_id=self.eos_id,
                             stops=self.stops, label="HF")

    def report(self) -> dict:
        kernels = dict(self.export.get("linear_attention_kernels") or {})
        if self.forced is not None:
            kernels["forced_torch_fallbacks"] = self.forced
        return {"load": {k: v for k, v in self.export.items() if k != "linear_attention_kernels"},
                "kernels": kernels,
                "kernel_path": kernels.get("kernel_path") or ("torch-fallback" if self.forced is not None else None),
                "placement": {"device": self.device, "devices": [{"cuda": 0, "role": "compute"}]
                              if self.device != "cpu" else []},
                "tokenizer": {"dir": self.tok_dir, "tokenizer_json_sha256": A.file_sha256(Path(self.tok_dir) / "tokenizer.json"),
                              "chat_template_sha256": A.file_sha256(Path(self.tok_dir) / "chat_template.jinja")},
                "thinking": self.thinking, "eos_id": self.eos_id, "stops": self.stops, "stop_names": self.stop_names,
                "ledger_band_mask": None}
