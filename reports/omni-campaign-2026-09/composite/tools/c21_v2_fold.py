#!/usr/bin/env python3
"""C21 -- the v2 fold-init: a v2 checkpoint whose every trunk tensor is a FOLD of our sources.

Ruling of record: cyril-v2-arch-and-fold-init-2026-10-21 ("[Cyril quote withheld]").  Plan:
V2_FOLD_INIT_PLAN_2026-10-21.md sections 3-4.  This build covers the TRUNK half of the plan:

    source  a Qwen3.5/3.8 HF text trunk (T = the run-1 full-KD student, the plan's default)
    target  any v2 profile whose per-head shapes equal the source's (co2-2b, co2-fold-proxy)

Per v2 layer k (sequential, dual stream: every fit reads the rows the already-folded v2 prefix
produces, never the source's rows):

    stream  z = alpha . h . C            C = c20's global orthonormal chart, alpha = the tied
                                         embedding's scale (below); one basis for the whole stream
    reads   W_v2 = (rho_l / s) . W_src diag(w_norm) C      the source norm weight folded in,
                                         rho_l = E[rms(z_k) / (alpha rms(h_l))] per pooled source
                                         layer, s = mean(rho) the v2 norm's scalar weight
    writes  prior = alpha . C^T W_src, then a prior-anchored ridge (c14.law_abiding_solve_gram)
                                         against the pooled source writes  sum_l write_l(h_l) . alpha C
    mixer   whole heads, selected by their write energy on z_k from the pooled source layers
            (c19 period map); every per-head tensor (conv taps, decay, A_log, gate norm, q/k norm)
            moves with its head
    ffn     union coreset over every pooled source channel: energy prefilter, then c14's pivoted
            Cholesky on the activation Gram of the survivors, then the ridge refit of `down`
    vocab   W_tied = H_src diag(w_final) C; alpha = median ||W[t]|| / ||E_src[t] C|| makes the
            tied input rows sit at the source stream's scale; control rows = mean folded text row

Every number a fit produces is a DIAGNOSTIC (owner policy 2026-10-06): relL2, retained energy,
update scale, merge loss.  Functional refusals are structural only: geometry mismatches, a
non-finite tensor, an assembled-model forward that disagrees with the fold's own stream, a
parameter the receipt cannot name a fold method for, and the >= 4x rule on selection components.

NOT in this build (recorded in the receipt as `not_built`): the K3 form-E union term (needs the
P1 refold of K3 rows on fit tokens, plan section 7 items 2-3), the media-unit channels, the raw
boundary and x0 head ridge fits (section 4.5).  Those tensors are installed as exact ZEROS
(method `zero_placeholder`, norms `unit`) so nothing random survives; a checkpoint with
placeholders is a mechanics artefact, never a candidate base.

Tests: test_c21_v2_fold.py (CPU; the identity-chart fold of a synthetic Qwen3.5 trunk reproduces
its logits, so every charting/transplant identity above is checked end to end).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

TOOL_NAME = "c21_v2_fold"
TOOL_VERSION = "1.2.0"   # 1.1.0 arm F-BE (--target-mode block_end); 1.2.0 arm F-FND (--final-norm diag)
RECEIPT_SCHEMA = "chimera_omni.composite.v2_fold_receipt.v1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKTREE_CHIMERA_SRC = ("C:/Users/cereal/Documents/ai training/.agent-worktrees/chimera-omni-5d4d4e1dcd/"
                        "workers/claude/arch-v2-20261021/src")
STUB_VOCAB = 8                 # the period models carry no vocabulary: 8 zero rows satisfy the importer
ASSEMBLY_TOLERANCE = 1.0e-3    # assembled-model forward vs the fold's own stream (a bug detector, not a fit gate)
FOLD_METHODS = ("chart", "chart_scalar", "head_select", "union_coreset", "ridge", "mean_rows",
                "zero_init_attnres", "zero_placeholder", "unit")


def _resolve_dir(env: str, candidates: list[str], probes: list[str], label: str) -> str:
    override = os.environ.get(env)
    options = ([override] if override else []) + candidates
    for cand in options:
        if cand and all(os.path.exists(os.path.join(cand, p)) for p in probes):
            return os.path.abspath(cand)
    raise SystemExit(f"{label} ({', '.join(probes)}) was not found in {options}; set {env}")


K3_LANE_DIR = _resolve_dir("K3_LANE_DIR", [os.path.join(ROOT, "k3", "harness"), os.path.join(ROOT, "k3")],
                           ["k3_common.py", "c14_union_coreset.py"], "the k3 lane (c14)")
CHIMERA_SRC = _resolve_dir("CHIMERA_SRC", [WORKTREE_CHIMERA_SRC],
                           [os.path.join("chimera_omni", "config.py")], "the chimera-omni arch-v2 source tree")
for _dir in (HERE, K3_LANE_DIR, CHIMERA_SRC):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

import c14_union_coreset as c14  # noqa: E402
import c19_v2_alignment as c19  # noqa: E402
import c20_stream_chart as c20  # noqa: E402
from chimera_omni.config import V2_PROFILES, ChimeraConfig  # noqa: E402
from chimera_omni.importing.profiles import qwen35_chimera_config  # noqa: E402
from chimera_omni.importing.qwen35 import QWEN_TEXT_PREFIX, import_qwen35_state  # noqa: E402
from chimera_omni.modeling import ChimeraOmniModel  # noqa: E402
from chimera_omni.modeling.layers import (  # noqa: E402
    GatedGQAAttention,
    KimiDeltaAttention,
    causal_valid_attention_mask,
    use_chunked_gdn_scan,
)


def _file_sha256_at_import(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# hashed when the module is imported, so a receipt names the code that RAN even if the file is replaced mid-run
TOOL_SHA256 = _file_sha256_at_import(__file__)
DEP_SHA256 = {name: _file_sha256_at_import(mod.__file__) for name, mod in (("c14", c14), ("c19", c19), ("c20", c20))}


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def numel(shape: Iterable[int]) -> int:
    return int(math.prod(int(v) for v in shape))


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(x.float().pow(2).mean(dim=-1) + eps)


def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Chimera's RMSNorm with a unit weight (the exact read every block applies before its weight)."""

    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# --------------------------------------------------------------------------
# the source trunk
# --------------------------------------------------------------------------
class QwenTrunkSource:
    """A Qwen3.5/3.8 HF text trunk, read tensor by tensor (a checkpoint directory or, in tests, a mapping).

    Layers are imported period by period through `importing.qwen35.import_qwen35_state` -- the
    exact importer, unchanged -- into a stub-vocabulary Chimera model, so the source forward the
    fold reads is the proven import and not a re-implementation.
    """

    def __init__(self, hf_config: dict[str, Any], getter: Callable[[str], torch.Tensor], names: Iterable[str], *,
                 label: str, path: str | None = None, prefix: str = QWEN_TEXT_PREFIX) -> None:
        self.hf_config = json.loads(json.dumps(hf_config))
        self.text = dict(self.hf_config.get("text_config", self.hf_config))
        self._get = getter
        self.names = set(names)
        self.label = label
        self.path = path
        self.prefix = prefix
        self.hidden = int(self.text["hidden_size"])
        self.num_layers = int(self.text["num_hidden_layers"])
        self.period = int(self.text.get("full_attention_interval") or 4)
        self.tied = bool(self.text.get("tie_word_embeddings", self.hf_config.get("tie_word_embeddings", False)))
        self.rms_eps = float(self.text["rms_norm_eps"])

    @classmethod
    def from_dir(cls, root: str | Path, *, label: str | None = None) -> "QwenTrunkSource":
        from safetensors import safe_open

        root = Path(root)
        hf_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        else:
            single = root / "model.safetensors"
            with safe_open(str(single), "pt") as handle:
                weight_map = {name: single.name for name in handle.keys()}
        handles: dict[str, Any] = {}

        def getter(name: str) -> torch.Tensor:
            shard = weight_map[name]
            if shard not in handles:
                handles[shard] = safe_open(str(root / shard), "pt")
            return handles[shard].get_tensor(name)

        return cls(hf_config, getter, weight_map.keys(), label=label or root.name, path=str(root))

    def get(self, name: str) -> torch.Tensor:
        if name not in self.names:
            raise KeyError(f"{self.label}: no tensor {name!r}")
        return self._get(name)

    def embed(self) -> torch.Tensor:
        return self.get(f"{self.prefix}embed_tokens.weight")

    def head(self) -> torch.Tensor:
        for name in ("lm_head.weight", f"{self.prefix}lm_head.weight", "model.lm_head.weight"):
            if name in self.names:
                return self.get(name)
        if not self.tied:
            raise KeyError(f"{self.label}: untied config but no lm_head")
        return self.embed()

    def final_norm(self) -> torch.Tensor:
        """Chimera's multiplicative weight: Qwen's (1 + w), in fp32 (as the importer stores it)."""

        return self.get(f"{self.prefix}norm.weight").float() + 1.0

    def layer_names(self, layer: int) -> list[str]:
        head = f"{self.prefix}layers.{layer}."
        return sorted(name for name in self.names if name.startswith(head))

    def period_config(self, first: int, count: int) -> ChimeraConfig:
        text = dict(self.text)
        text["num_hidden_layers"] = int(count)
        text["vocab_size"] = STUB_VOCAB
        if isinstance(text.get("layer_types"), list):
            text["layer_types"] = text["layer_types"][first:first + count]
        hf = dict(self.hf_config)
        if "text_config" in hf:
            hf["text_config"] = text
        else:
            hf = text
        return qwen35_chimera_config(hf, ledger_bands=(("control", STUB_VOCAB),),
                                     allow_unverified_output_gate=True)

    def period_model(self, first: int, count: int, device: torch.device) -> tuple[ChimeraOmniModel, dict[str, Any]]:
        if first % self.period or count % self.period:
            raise ValueError(f"a period model must start on and span whole periods of {self.period}")
        cfg = self.period_config(first, count)
        state: dict[str, torch.Tensor] = {}
        for local in range(count):
            src_head = f"{self.prefix}layers.{first + local}."
            dst_head = f"{self.prefix}layers.{local}."
            names = self.layer_names(first + local)
            if not names:
                raise KeyError(f"{self.label}: layer {first + local} has no tensors")
            for name in names:
                state[dst_head + name[len(src_head):]] = self.get(name)
        state[f"{self.prefix}embed_tokens.weight"] = torch.zeros(STUB_VOCAB, self.hidden)
        state[f"{self.prefix}norm.weight"] = self.get(f"{self.prefix}norm.weight")
        if not cfg.tie_word_embeddings:
            state["lm_head.weight"] = torch.zeros(STUB_VOCAB, self.hidden)
        with torch.device(device):
            model, receipt = import_qwen35_state(state, cfg, device=str(device))
        model.eval()
        return model, {"layers": [first, first + count - 1], "consumed_tensors": receipt["consumed_tensors"],
                       "trunk_imported": receipt["parameter_counts"]["trunk_imported"]}


def block_state(block: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A CPU copy of one imported block, bf16 wherever that is lossless (every bf16-sourced matrix)."""

    out: dict[str, torch.Tensor] = {}
    for name, tensor in block.state_dict().items():
        t = tensor.detach().float().cpu()
        if t.numel() >= 65536:
            low = t.to(torch.bfloat16)
            if torch.equal(low.float(), t):
                out[name] = low
                continue
        out[name] = t.clone()
    return out


# --------------------------------------------------------------------------
# calibration rows
# --------------------------------------------------------------------------
@dataclass
class Calibration:
    input_ids: torch.Tensor          # [N, S] long
    valid: torch.Tensor              # [N, S] bool
    fit_rows: list[int]              # rows every fit reads (inner + check)
    check_rows: list[int]            # the ridge's internal lambda-selection split (a subset of fit_rows)
    held_rows: list[int]             # never fitted: diagnostics and the behaviour checks
    record: dict[str, Any] = field(default_factory=dict)

    def mask(self, rows: Iterable[int]) -> torch.Tensor:
        m = torch.zeros(self.input_ids.shape[0], dtype=torch.bool)
        m[list(rows)] = True
        return self.valid & m.unsqueeze(1)

    @property
    def inner_rows(self) -> list[int]:
        return [r for r in self.fit_rows if r not in set(self.check_rows)]


def load_corpus(path: str | Path, *, rows: int | None = None, seq_len: int | None = None,
                check_every: int = 6) -> Calibration:
    """The omni evidence corpus (`qwen_input_ids.safetensors`: split 0 = fit row, 1 = held row)."""

    from safetensors.torch import load_file

    t = load_file(str(path))
    ids, lengths, split = t["input_ids"].long(), t["real_lengths"].long(), t["split"].long()
    if rows is not None:
        # keep the fit/held pattern: take whole rows from the front
        ids, lengths, split = ids[:rows], lengths[:rows], split[:rows]
    if seq_len is not None:
        ids = ids[:, :seq_len]
        lengths = lengths.clamp(max=seq_len)
    valid = torch.arange(ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    fit = [i for i in range(ids.shape[0]) if int(split[i]) == 0]
    held = [i for i in range(ids.shape[0]) if int(split[i]) == 1]
    if not fit or not held:
        raise SystemExit(f"{path}: need fit (split 0) and held (split 1) rows; have {len(fit)} / {len(held)}")
    check = fit[check_every - 1::check_every] or fit[-1:]
    record = {"path": str(path), "sha256": sha256_file(path), "rows": int(ids.shape[0]), "seq_len": int(ids.shape[1]),
              "fit_rows": fit, "check_rows": check, "held_rows": held,
              "fit_tokens": int(valid[fit].sum()), "held_tokens": int(valid[held].sum()),
              "rule": "split 0 = fit, split 1 = held (whole rows, never fitted); check = every "
                      f"{check_every}th fit row (the ridge's internal lambda split)"}
    return Calibration(ids, valid, fit, check, held, record)


# --------------------------------------------------------------------------
# small fold primitives (everything here is exact linear algebra)
# --------------------------------------------------------------------------
def read_rows(weight: torch.Tensor, norm_weight: torch.Tensor, chart: torch.Tensor, scale: float) -> torch.Tensor:
    """A pre-norm read W [out x H] of the source stream, charted: scale . W diag(w) C  [out x d]."""

    return scale * ((weight.float() * norm_weight.float().unsqueeze(0)) @ chart)


def write_cols(weight: torch.Tensor, chart: torch.Tensor, alpha: float) -> torch.Tensor:
    """A write W [H x in] into the source stream, charted: alpha . C^T W  [d x in]."""

    return alpha * (chart.T @ weight.float())


class GramPack:
    """X^T X, X^T Y, ||Y||^2 on the inner, check and held splits (fp64), for one ridge."""

    def __init__(self, width: int, out: int, device: torch.device) -> None:
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)  # noqa: E731
        self.g = {k: z(width, width) for k in ("inner", "check", "held")}
        self.b = {k: z(width, out) for k in ("inner", "check", "held")}
        self.y = {k: 0.0 for k in ("inner", "check", "held")}
        self.n = {k: 0 for k in ("inner", "check", "held")}

    def add(self, split: str, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[0] == 0:
            return
        x, y = x.double(), y.double()       # fp64 products: the refit Gram is what the ridge conditions on
        self.g[split] += x.T @ x
        self.b[split] += x.T @ y
        self.y[split] += float(y.double().pow(2).sum())
        self.n[split] += int(x.shape[0])

    def rel_l2(self, split: str, weight: torch.Tensor) -> float | None:
        """||X W^T - Y|| / ||Y|| from the Grams alone (c14's _check_sse form)."""

        if self.n[split] == 0 or self.y[split] <= 0:
            return None
        sse = c14._check_sse(weight.to(self.g[split].device), self.g[split], self.b[split], self.y[split])
        return math.sqrt(max(sse, 0.0) / self.y[split])

    def solve(self, prior: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """c14.law_abiding_solve_gram on (inner, check); fit/held relL2 and the prior's held relL2 recorded."""

        prior = prior.double().cpu()
        g_inner, b_inner = self.g["inner"].cpu().clone(), self.b["inner"].cpu().clone()
        g_check, b_check = self.g["check"].cpu().clone(), self.b["check"].cpu().clone()
        fit_g = (self.g["inner"] + self.g["check"]).cpu()
        fit_b = (self.b["inner"] + self.b["check"]).cpu()
        fit_y = self.y["inner"] + self.y["check"]
        law: dict[str, Any]
        try:
            weight, law = c14.law_abiding_solve_gram(g_inner, b_inner, g_check, b_check, self.y["check"], prior)
            law = {k: v for k, v in law.items() if k != "ladder"} | {"ladder_rungs": len(law.get("ladder", []))}
        except (SystemExit, RuntimeError) as exc:
            # owner policy 2026-10-06: a numeric veto (or a ladder Cholesky that failed) is recorded,
            # never a stopped layer; the most prior-anchored rung is installed and the record says so.
            try:
                weight = c14.prior_anchored_gram(fit_g, fit_b, prior, rel=c14.LAMBDA_LADDER[-1])
                installed = f"lambda_relative {c14.LAMBDA_LADDER[-1]} (recorded, not gated)"
            except RuntimeError:
                weight, installed = prior.clone(), "the charted prior (the Gram was singular at every rung)"
            law = {"law_refused": str(exc), "installed": installed}
        fit_rel = (math.sqrt(max(c14._check_sse(weight, fit_g, fit_b, fit_y), 0.0) / fit_y) if fit_y > 0 else None)
        record = {**law, "fit_rel_l2": fit_rel, "held_rel_l2": self.rel_l2("held", weight),
                  "prior_held_rel_l2": self.rel_l2("held", prior), "rows": dict(self.n)}
        return weight.float(), record


def split_of(row: int, calib: Calibration) -> str | None:
    if row in calib.held_rows:
        return "held"
    if row in calib.check_rows:
        return "check"
    if row in calib.fit_rows:
        return "inner"
    return None


def masked_rows(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return x[valid]


def pivoted_cholesky_select(gram: torch.Tensor, k: int) -> torch.Tensor:
    """c14's pivoted-Cholesky column selection (imported, not copied)."""

    return c14.pivoted_columns_gram(gram.double().cpu(), k)


# --------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------
@dataclass
class FoldOptions:
    device: str = "cpu"
    batch: int = 1
    prefilter_factor: int = 4
    gdn_chunk: int = 64
    tie: str = "head"                  # head (plan default) | joint (arm F-TIEJ)
    target_mode: str = "writes"        # writes (plan default) | block_end (arm F-BE: the last FFN of each depth
                                       # block refits against alpha h_end C - z, what the block still has to write)
    chart: torch.Tensor | None = None  # precomputed c20 chart (skips pass 1)
    require_4x: bool = True            # the 4x rule; only the identity-fold mechanics test turns it off
    chart_record: dict[str, Any] | None = None
    save_chart: str | None = None      # write the pass-1 chart as soon as it exists (other arms reuse it)
    final_norm: str = "scalar"         # scalar (plan 4.5) | diag (arm F-FND: per-channel closed-form fit of the logit read)


@dataclass
class SourceLayerRecord:
    kind: str
    state: dict[str, torch.Tensor]
    mix: torch.Tensor          # [N, S, d] alpha . a_l C  (CPU fp32)
    ffn: torch.Tensor          # [N, S, d] alpha . f_l C
    rms_in: torch.Tensor       # [N, S] rms(h_l)      (mixer input)
    rms_mid: torch.Tensor      # [N, S] rms(h_l + a_l) (FFN input)


class Fold:
    def __init__(self, source: QwenTrunkSource, target_cfg: ChimeraConfig, calib: Calibration,
                 options: FoldOptions) -> None:
        self.src = source
        self.cfg = target_cfg
        self.calib = calib
        self.opt = options
        self.device = torch.device(options.device)
        self.d = int(target_cfg.hidden_size)
        self.eps = float(target_cfg.rms_norm_eps)
        self.components: list[dict[str, Any]] = []
        self.layers: list[dict[str, Any]] = []
        self.timings: dict[str, float] = {}
        self.param_methods: dict[str, str] = {}
        self._check_geometry()
        kinds = tuple(target_cfg.block_kind(i) for i in range(int(source.period)))
        self.alignment = c19.alignment_document(
            source_name=source.label, source_layers=source.num_layers,
            target_profile=str(target_cfg.metadata.get("profile", "custom")), target_layers=int(target_cfg.num_layers),
            period=source.period, kinds=kinds)
        self.alignment_sha256 = c19.document_sha256(self.alignment)

    # ---------------------------------------------------------------- geometry
    def _check_geometry(self) -> None:
        cfg, text = self.cfg, self.src.text
        period = self.src.period
        if tuple(cfg.block_pattern) != tuple(["kda"] * (period - 1) + ["gated_gqa"]):
            raise SystemExit(f"target block pattern {cfg.block_pattern} is not the source's {period}-period")
        if cfg.residual_mode != "block_attnres" or cfg.attnres_init != "zero":
            raise SystemExit("the fold relies on zero-init Block AttnRes (uniform mix == plain residual under pre-norm)")
        need = {
            "attention_head_dim": (cfg.resolved_attention_head_dim(), int(text["head_dim"])),
            "kda_head_dim": (cfg.resolved_kda_head_dim(), int(text["linear_key_head_dim"])),
            "kda_value_head_dim": (cfg.resolved_kda_value_head_dim(), int(text["linear_value_head_dim"])),
            "kda_short_conv_kernel_size": (int(cfg.kda_short_conv_kernel_size), int(text["linear_conv_kernel_dim"])),
            "text_vocab": (int(cfg.ledger_base_vocab_size), int(text["vocab_size"])),
        }
        bad = {k: v for k, v in need.items() if v[0] != v[1]}
        if bad:
            raise SystemExit(f"whole-head transplant needs equal per-head shapes; target != source on {bad}")
        if cfg.resolved_kda_num_heads() != cfg.resolved_kda_num_key_heads():
            raise SystemExit("this build folds into 1:1 GDN key heads only (co2-2b, co2-fold-proxy); grouped "
                             "target keys (co2-4b) need a shared-key selection rule first")
        if cfg.kda_decay_kind != "gdn" or cfg.kda_output_gate != "silu" or cfg.activation_kind != "silu":
            raise SystemExit("target must be the GDN / silu-gate / SiLU-GLU v2 trunk")
        if not cfg.tie_word_embeddings:
            raise SystemExit("v2 ties its vocabulary; an untied target is not a v2 profile")
        src_rope = self.src.period_config(0, period)
        if (int(cfg.gqa_rope_dim), float(cfg.gqa_rope_theta)) != (int(src_rope.gqa_rope_dim), float(src_rope.gqa_rope_theta)):
            raise SystemExit(f"rotary differs: target ({cfg.gqa_rope_dim}, {cfg.gqa_rope_theta}) vs source "
                             f"({src_rope.gqa_rope_dim}, {src_rope.gqa_rope_theta}); heads would not move exactly")
        if int(cfg.num_layers) > self.src.num_layers:
            raise SystemExit("a fold cannot be deeper than its source")
        self.src_period_cfg = src_rope

    # ---------------------------------------------------------------- source forward
    def _run_source_period(self, model: ChimeraOmniModel, h: torch.Tensor,
                           visit: Callable[[int, torch.nn.Module, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
                           first: int) -> None:
        """One imported period over every calibration row, in place on h; `visit` sees each layer's rows."""

        valid = self.calib.valid.to(self.device)
        n = h.shape[0]
        for local, block in enumerate(model.blocks):
            layer = first + local
            for start in range(0, n, self.opt.batch):
                rows = slice(start, min(n, start + self.opt.batch))
                x = h[rows]
                v = valid[rows]
                with torch.no_grad():
                    a = block.attention_output(x, token_valid_mask=v)
                    xm = x + a
                    f = block.feed_forward_output(xm, v)
                    visit(layer, block, start, x, a, xm, f)
                    h[rows] = torch.where(v.unsqueeze(-1), xm + f, torch.zeros_like(x))

    def _source_embed(self) -> torch.Tensor:
        embed = self.src.embed()
        ids = self.calib.input_ids
        h = embed[ids.reshape(-1)].float().reshape(*ids.shape, -1)
        return torch.where(self.calib.valid.unsqueeze(-1), h, torch.zeros_like(h)).to(self.device)

    def pass1_chart(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """c20: the uncentred, RMS-normalised second moment of every source layer's rows on fit tokens."""

        t0 = time.time()
        moment = c20.SecondMoment(self.src.hidden, device=self.device)
        fit = self.calib.mask(self.calib.fit_rows).to(self.device)
        h = self._source_embed()
        for start in range(0, h.shape[0], self.opt.batch):
            rows = slice(start, min(h.shape[0], start + self.opt.batch))
            moment.add(h[rows][fit[rows]], layer=-1, eps=self.src.rms_eps)

        def visit(layer, block, start, x, a, xm, f):
            m = fit[start:start + x.shape[0]]
            moment.add(xm[m], layer=layer, eps=self.src.rms_eps)

        for first in range(0, self.src.num_layers, self.src.period):
            model, _ = self.src.period_model(first, self.src.period, self.device)
            use_chunked_gdn_scan(model, self.opt.gdn_chunk)
            self._run_source_period(model, h, visit, first)
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            log(f"pass 1: source layers {first}-{first + self.src.period - 1} accumulated ({time.time() - t0:.0f}s)")
        last = self.src.num_layers
        for start in range(0, h.shape[0], self.opt.batch):
            rows = slice(start, min(h.shape[0], start + self.opt.batch))
            moment.add(h[rows][fit[rows]], layer=last, eps=self.src.rms_eps)
        chart, record = c20.fit_chart(moment, self.d)
        record["pooled_rows"] = "source embedding rows + every layer's post-attention (FFN-input) rows + final rows, fit tokens"
        self.timings["pass1_chart"] = time.time() - t0
        return chart.float(), record

    # ---------------------------------------------------------------- vocabulary
    def fold_vocabulary(self, model: ChimeraOmniModel, chart: torch.Tensor) -> float:
        """W = H diag(w_final) C (head side, plan default) and the stream scale alpha."""

        t0 = time.time()
        base = int(self.cfg.ledger_base_vocab_size)
        C = chart.to(self.device)
        w_final = self.src.final_norm().to(self.device)
        head = self.src.head()
        embed = self.src.embed()
        W = torch.empty(base, self.d, dtype=torch.float32, device=self.device)
        EC = torch.empty(base, self.d, dtype=torch.float32, device=self.device)
        for start in range(0, base, 16384):
            rows = slice(start, min(base, start + 16384))
            W[rows] = (head[rows].to(self.device).float() * w_final) @ C
            EC[rows] = embed[rows].to(self.device).float() @ C
        fit = self.calib.mask(self.calib.fit_rows)
        tokens = torch.unique(self.calib.input_ids[fit]).to(self.device)
        ratio = W[tokens].norm(dim=-1) / EC[tokens].norm(dim=-1).clamp_min(1e-30)
        alpha = float(ratio.median())
        cos = F.cosine_similarity(W[tokens], alpha * EC[tokens], dim=-1)
        if self.opt.tie == "joint":
            # arm F-TIEJ: the tied rows split the difference between the two charted source matrices.
            W = 0.5 * (W + alpha * EC)
        elif self.opt.tie != "head":
            raise SystemExit(f"unknown --tie {self.opt.tie!r}")
        with torch.no_grad():
            model.embed.weight[:base].copy_(W)
            model.embed.weight[base:].copy_(W.mean(dim=0, keepdim=True).expand(int(self.cfg.vocab_size) - base, -1))
        self.param_methods["embed.weight"] = "chart"
        self.param_methods["lm_head.weight"] = "chart"
        self.vocab_record = {
            "tie": self.opt.tie,
            "rule": ("W = H diag(w_final) C" if self.opt.tie == "head" else "W = (H diag(w_final) C + alpha E C) / 2"),
            "control_rows": "mean of the folded text rows (spec section 9)",
            "alpha": alpha,
            "alpha_rule": "median over fit-corpus tokens of ||W[t]|| / ||E[t] C|| (tied input rows at the source stream's scale)",
            "alpha_ratio_quartiles": [float(q) for q in torch.quantile(ratio.float(), torch.tensor([0.25, 0.5, 0.75], device=ratio.device))],
            "input_side_cos_quartiles": [float(q) for q in torch.quantile(cos.float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=cos.device))],
            "tokens_measured": int(tokens.numel()),
        }
        candidate = numel(head.shape) + (0 if self.src.tied else numel(embed.shape))
        self._component(None, "embed", "chart", candidate, numel(model.embed.weight.shape),
                        diagnostics={k: self.vocab_record[k] for k in ("alpha", "input_side_cos_quartiles")},
                        sources=[{"source": self.src.label, "tensors": ["lm_head", "final norm"] + ([] if self.src.tied else ["embed_tokens"])}])
        del W, EC
        self.timings["vocabulary"] = time.time() - t0
        return alpha

    # ---------------------------------------------------------------- bookkeeping
    def _component(self, layer: int | None, component: str, method: str, candidate: int, installed: int, *,
                   diagnostics: dict[str, Any] | None = None, sources: list[dict[str, Any]] | None = None,
                   selection: bool = False) -> None:
        if method not in FOLD_METHODS:
            raise SystemExit(f"{component}: {method!r} is not a fold method")
        compression = (candidate / installed) if installed else None
        row = {"target_layer": layer, "component": component, "method": method,
               "candidate_params": int(candidate), "installed_params": int(installed),
               "compression": compression, "selection_component": selection,
               "clears_4x": None if compression is None else bool(compression >= 4.0),
               "diagnostics": diagnostics or {}, "sources": sources or []}
        self.components.append(row)

    # ---------------------------------------------------------------- pass 2 source writes
    def collect_group(self, periods: list[int], h: torch.Tensor, C: torch.Tensor, alpha: float) -> dict[int, SourceLayerRecord]:
        records: dict[int, SourceLayerRecord] = {}
        n, s = self.calib.input_ids.shape
        buffers: dict[int, dict[str, torch.Tensor]] = {}

        def visit(layer, block, start, x, a, xm, f):
            if layer not in buffers:
                buffers[layer] = {"mix": torch.zeros(n, s, self.d), "ffn": torch.zeros(n, s, self.d),
                                  "rms_in": torch.zeros(n, s), "rms_mid": torch.zeros(n, s)}
            rows = slice(start, start + x.shape[0])
            buf = buffers[layer]
            buf["mix"][rows] = (alpha * (a @ C)).cpu()
            buf["ffn"][rows] = (alpha * (f @ C)).cpu()
            buf["rms_in"][rows] = rms(x, self.src.rms_eps).cpu()
            buf["rms_mid"][rows] = rms(xm, self.src.rms_eps).cpu()
            # per-source-layer retained energy of the stream under C (held rows, a diagnostic)
            if block is not None:
                held = self.calib.mask(self.calib.held_rows)[rows].to(self.device)
                if bool(held.any()):
                    xs = xm[held]
                    proj = (xs @ C) @ C.T
                    acc = self._retained.setdefault(layer, [0.0, 0.0])
                    acc[0] += float((xs - proj).pow(2).sum())
                    acc[1] += float(xs.pow(2).sum())

        for q in periods:
            first = q * self.src.period
            model, info = self.src.period_model(first, self.src.period, self.device)
            use_chunked_gdn_scan(model, self.opt.gdn_chunk)
            self._run_source_period(model, h, visit, first)
            for local, block in enumerate(model.blocks):
                layer = first + local
                buf = buffers[layer]
                records[layer] = SourceLayerRecord(block.kind, block_state(block), buf["mix"], buf["ffn"],
                                                   buf["rms_in"], buf["rms_mid"])
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        # the source stream at the group end, charted: the F-BE target (and the period-end diagnostic)
        self._group_end = torch.zeros(n, s, self.d)
        for start in range(0, n, self.opt.batch):
            rows = slice(start, min(n, start + self.opt.batch))
            self._group_end[rows] = (alpha * (h[rows] @ C)).cpu()
        return records

    # ---------------------------------------------------------------- v2 helpers
    def _v2_batches(self) -> Iterable[tuple[slice, torch.Tensor]]:
        valid = self.calib.valid
        n = valid.shape[0]
        for start in range(0, n, self.opt.batch):
            rows = slice(start, min(n, start + self.opt.batch))
            yield rows, valid[rows].to(self.device)

    def _row_splits(self, rows: slice) -> list[str | None]:
        return [split_of(r, self.calib) for r in range(rows.start, rows.stop)]

    def _rho(self, z_rms: torch.Tensor, src_rms: torch.Tensor, alpha: float) -> float:
        """E[rms(z) / (alpha rms(h_l))] over fit tokens: the scale the folded read needs."""

        fit = self.calib.mask(self.calib.fit_rows)
        ratio = z_rms[fit] / (alpha * src_rms[fit]).clamp_min(1e-30)
        return float(ratio.mean())

    def _src_cfg(self, kind: str) -> ChimeraConfig:
        return dataclasses.replace(self.src_period_cfg, hidden_size=self.d)

    # ---------------------------------------------------------------- mixers
    def _gdn_candidate(self, st: dict[str, torch.Tensor], C: torch.Tensor, rho: float, alpha: float) -> KimiDeltaAttention:
        cfg = self._src_cfg("kda")
        with torch.device(self.device):
            m = KimiDeltaAttention(cfg)
        use_chunked_gdn_scan(m, self.opt.gdn_chunk)
        w1 = st["norm1.weight"].to(self.device)
        g = lambda name: st[name].to(self.device)  # noqa: E731
        with torch.no_grad():
            for name in ("q_proj", "k_proj", "v_proj", "output_gate", "write_strength", "decay_down"):
                getattr(m, name).weight.copy_(read_rows(g(f"mixer.{name}.weight"), w1, C, rho))
            for conv in ("q_conv", "k_conv", "v_conv"):
                getattr(m, conv).conv.weight.copy_(g(f"mixer.{conv}.conv.weight"))
            m.decay_bias.copy_(g("mixer.decay_bias"))
            m.decay_log_scale.copy_(g("mixer.decay_log_scale"))
            m.output_norm.weight.copy_(g("mixer.output_norm.weight"))
            m.out.weight.copy_(write_cols(g("mixer.out.weight"), C, alpha))
        return m.eval()

    def _gqa_candidate(self, st: dict[str, torch.Tensor], C: torch.Tensor, rho: float, alpha: float) -> GatedGQAAttention:
        cfg = self._src_cfg("gated_gqa")
        with torch.device(self.device):
            m = GatedGQAAttention(cfg)
        w1 = st["norm1.weight"].to(self.device)
        g = lambda name: st[name].to(self.device)  # noqa: E731
        with torch.no_grad():
            for name in ("q_proj", "gate_proj", "k_proj", "v_proj"):
                getattr(m, name).weight.copy_(read_rows(g(f"mixer.{name}.weight"), w1, C, rho))
            m.q_norm.weight.copy_(g("mixer.q_norm.weight"))
            m.k_norm.weight.copy_(g("mixer.k_norm.weight"))
            m.out.weight.copy_(write_cols(g("mixer.out.weight"), C, alpha))
        return m.eval()

    def _run_mixer(self, m: torch.nn.Module, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The mixer on normalised rows u exactly as ChimeraBlock.attention_output calls it; returns pre-out rows."""

        captured: list[torch.Tensor] = []
        hook = m.out.register_forward_hook(lambda mod, inp, out: captured.append(inp[0]))
        try:
            x = torch.where(v.unsqueeze(-1), u, torch.zeros_like(u))
            if isinstance(m, GatedGQAAttention):
                m(x, attention_mask=causal_valid_attention_mask(v), token_valid_mask=v)
            else:
                m(x, token_valid_mask=v)
        finally:
            hook.remove()
        y = captured[0]
        return torch.where(v.unsqueeze(-1), y, torch.zeros_like(y))

    def _head_energies(self, m: torch.nn.Module, z: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
        """Per-head write energy on fit tokens: sum_t ||O_h y_h(t)||^2 = <G_h, O_h^T O_h>."""

        out = m.out.weight.detach().view(self.d, heads, head_dim)
        ogram = torch.einsum("dhi,dhj->hij", out, out)
        gram = torch.zeros(heads, head_dim, head_dim, dtype=torch.float64, device=self.device)
        for rows, v in self._v2_batches():
            fitrows = torch.tensor([s in ("inner", "check") for s in self._row_splits(rows)], device=self.device)
            mask = v & fitrows.unsqueeze(1)
            if not bool(mask.any()):
                continue
            with torch.no_grad():
                y = self._run_mixer(m, rms_normalize(z[rows], self.eps), v)[mask].view(-1, heads, head_dim)
                gram += torch.einsum("nhi,nhj->hij", y, y).double()
        return (gram * ogram.double()).sum(dim=(1, 2)).cpu()

    def fold_mixer(self, k: int, block: torch.nn.Module, pool: list[int], recs: dict[int, SourceLayerRecord],
                   z: torch.Tensor, C: torch.Tensor, alpha: float) -> dict[str, Any]:
        kind = block.kind
        z_rms = rms(z, self.eps).cpu()
        rho = {l: self._rho(z_rms, recs[l].rms_in, alpha) for l in pool}
        sbar = float(sum(rho.values()) / len(rho))
        mixer = block.mixer
        cand_params = sum(sum(numel(t.shape) for n, t in recs[l].state.items() if n.startswith("mixer.")) for l in pool)
        if kind == "kda":
            hv, dk, dv = mixer.num_heads, mixer.key_head_dim, mixer.value_head_dim
            src_hv = int(self.src.text["linear_num_value_heads"])
            src_hk = int(self.src.text["linear_num_key_heads"])
            group = src_hv // src_hk
            energies = []
            for l in pool:
                cand = self._gdn_candidate(recs[l].state, C, rho[l], alpha)
                e = self._head_energies(cand, z, src_hv, dv)
                energies += [(float(e[h]), l, h) for h in range(src_hv)]
                del cand
            chosen = sorted(sorted(energies, reverse=True)[:hv], key=lambda t: (pool.index(t[1]), t[2]))
            prior = torch.zeros(self.d, hv * dv)
            with torch.no_grad():
                for i, (_, l, h) in enumerate(chosen):
                    st, r = recs[l].state, rho[l] / sbar
                    w1 = st["norm1.weight"]
                    kh = h // group
                    ks, vs, ds = slice(kh * dk, (kh + 1) * dk), slice(h * dv, (h + 1) * dv), slice(i * dk, (i + 1) * dk)
                    vi = slice(i * dv, (i + 1) * dv)
                    mixer.q_proj.weight[ds] = read_rows(st["mixer.q_proj.weight"][ks], w1, C.cpu(), r).to(self.device)
                    mixer.k_proj.weight[ds] = read_rows(st["mixer.k_proj.weight"][ks], w1, C.cpu(), r).to(self.device)
                    mixer.v_proj.weight[vi] = read_rows(st["mixer.v_proj.weight"][vs], w1, C.cpu(), r).to(self.device)
                    mixer.output_gate.weight[vi] = read_rows(st["mixer.output_gate.weight"][vs], w1, C.cpu(), r).to(self.device)
                    mixer.write_strength.weight[i] = read_rows(st["mixer.write_strength.weight"][h:h + 1], w1, C.cpu(), r)[0].to(self.device)
                    mixer.decay_down.weight[i] = read_rows(st["mixer.decay_down.weight"][h:h + 1], w1, C.cpu(), r)[0].to(self.device)
                    mixer.q_conv.conv.weight[ds] = st["mixer.q_conv.conv.weight"][ks].float().to(self.device)
                    mixer.k_conv.conv.weight[ds] = st["mixer.k_conv.conv.weight"][ks].float().to(self.device)
                    mixer.v_conv.conv.weight[vi] = st["mixer.v_conv.conv.weight"][vs].float().to(self.device)
                    mixer.decay_bias[i] = st["mixer.decay_bias"][h].float()
                    mixer.decay_log_scale[i] = st["mixer.decay_log_scale"][h].float()
                    mixer.output_norm.weight[i] = st["mixer.output_norm.weight"][h].float().to(self.device)
                    prior[:, vi] = write_cols(st["mixer.out.weight"][:, vs], C.cpu(), alpha)
            selection = [{"source_layer": l, "head": h, "key_head": h // group, "energy": e} for e, l, h in chosen]
            pool_heads = len(pool) * src_hv
            norm_detail = {}
        else:
            hq, hkv, hd = mixer.num_heads, mixer.num_kv_heads, mixer.head_dim
            qpg = hq // hkv
            src_hq = int(self.src.text["num_attention_heads"])
            src_hkv = int(self.src.text["num_key_value_heads"])
            src_qpg = src_hq // src_hkv
            if src_qpg < qpg:
                raise SystemExit(f"layer {k}: a v2 kv group needs {qpg} q heads, a source group has {src_qpg}")
            per_head: dict[tuple[int, int], float] = {}
            for l in pool:
                cand = self._gqa_candidate(recs[l].state, C, rho[l], alpha)
                e = self._head_energies(cand, z, src_hq, hd)
                for h in range(src_hq):
                    per_head[(l, h)] = float(e[h])
                del cand
            groups = sorted(((sum(per_head[(l, g * src_qpg + t)] for t in range(src_qpg)), l, g)
                             for l in pool for g in range(src_hkv)), reverse=True)[:hkv]
            groups = sorted(groups, key=lambda t: (pool.index(t[1]), t[2]))
            prior = torch.zeros(self.d, hq * hd)
            chosen = []
            q_norms, k_norms, weights = [], [], []
            with torch.no_grad():
                for gi, (gen, l, g) in enumerate(groups):
                    st, r = recs[l].state, rho[l] / sbar
                    w1 = st["norm1.weight"]
                    kv = slice(g * hd, (g + 1) * hd)
                    gs = slice(gi * hd, (gi + 1) * hd)
                    mixer.k_proj.weight[gs] = read_rows(st["mixer.k_proj.weight"][kv], w1, C.cpu(), r).to(self.device)
                    mixer.v_proj.weight[gs] = read_rows(st["mixer.v_proj.weight"][kv], w1, C.cpu(), r).to(self.device)
                    heads = sorted(sorted(((per_head[(l, g * src_qpg + t)], g * src_qpg + t) for t in range(src_qpg)),
                                          reverse=True)[:qpg], key=lambda t: t[1])
                    for t, (he, h) in enumerate(heads):
                        i = gi * qpg + t
                        qs, vi = slice(h * hd, (h + 1) * hd), slice(i * hd, (i + 1) * hd)
                        mixer.q_proj.weight[vi] = read_rows(st["mixer.q_proj.weight"][qs], w1, C.cpu(), r).to(self.device)
                        mixer.gate_proj.weight[vi] = read_rows(st["mixer.gate_proj.weight"][qs], w1, C.cpu(), r).to(self.device)
                        prior[:, vi] = write_cols(st["mixer.out.weight"][:, qs], C.cpu(), alpha)
                        chosen.append({"source_layer": l, "kv_group": g, "head": h, "energy": he})
                    q_norms.append(st["mixer.q_norm.weight"].float())
                    k_norms.append(st["mixer.k_norm.weight"].float())
                    weights.append(max(gen, 1e-30))
                wsum = sum(weights)
                qn = sum(w * t for w, t in zip(weights, q_norms)) / wsum
                kn = sum(w * t for w, t in zip(weights, k_norms)) / wsum
                mixer.q_norm.weight.copy_(qn.to(self.device))
                mixer.k_norm.weight.copy_(kn.to(self.device))
            spread = lambda ts, mean: max(float((t - mean).norm() / mean.norm().clamp_min(1e-30)) for t in ts)  # noqa: E731
            norm_detail = {"qk_norm_rule": "energy-weighted mean of the selected groups' source q/k norm weights "
                                           "(one v2 weight per layer; exact when every group comes from one source layer)",
                           "q_norm_max_rel_spread": spread(q_norms, qn), "k_norm_max_rel_spread": spread(k_norms, kn),
                           "source_layers_used": sorted({c["source_layer"] for c in chosen})}
            selection = chosen
            pool_heads = len(pool) * src_hq
        with torch.no_grad():
            block.norm1.weight.fill_(sbar)
        self.param_methods.update({f"blocks.{k}.norm1.weight": "chart_scalar"})
        for name, _ in mixer.named_parameters():
            self.param_methods[f"blocks.{k}.mixer.{name}"] = "ridge" if name == "out.weight" else "head_select"

        # the v2 mixer's own pre-out rows: exactness self-check against the candidates (batch 0),
        # then the ridge of `out` against the pooled source writes
        width = mixer.out.weight.shape[1]
        pack = GramPack(width, self.d, self.device)
        for rows, v in self._v2_batches():
            with torch.no_grad():
                y = self._run_mixer(mixer, rms_normalize(z[rows], self.eps) * sbar, v)
            target = sum(recs[l].mix[rows] for l in pool).to(self.device)
            for local, split in enumerate(self._row_splits(rows)):
                if split is None:
                    continue
                m = v[local]
                pack.add(split, y[local][m].float(), target[local][m].float())
        weight, ridge = pack.solve(prior)
        with torch.no_grad():
            mixer.out.weight.copy_(weight.to(self.device))
        installed = sum(numel(p.shape) for p in mixer.parameters())
        self._component(k, f"mixer.{kind}", "head_select", cand_params, installed, selection=True,
                        diagnostics={"rho": {str(l): v for l, v in rho.items()}, "norm_scalar": sbar,
                                     "pool_heads": pool_heads, "selected_heads": len(selection), **norm_detail,
                                     "out_ridge": ridge},
                        sources=[{"source_layer": l, "heads": sum(1 for c in selection if c["source_layer"] == l)}
                                 for l in pool])
        return {"kind": kind, "selection": selection, "rho": rho, "norm_scalar": sbar, "out_ridge": ridge}

    # ---------------------------------------------------------------- ffn
    def fold_ffn(self, k: int, block: torch.nn.Module, pool: list[int], recs: dict[int, SourceLayerRecord],
                 z: torch.Tensor, C: torch.Tensor, alpha: float, block_end: bool = False) -> dict[str, Any]:
        ffn = block.ffn
        kk = int(ffn.down.weight.shape[1])
        z_rms = rms(z, self.eps).cpu()
        rho = {l: self._rho(z_rms, recs[l].rms_mid, alpha) for l in pool}
        sbar = float(sum(rho.values()) / len(rho))
        gates, ups, downs, origin = [], [], [], []
        for l in pool:
            st = recs[l].state
            fused = st["ffn.up.weight"]
            width = fused.shape[0] // 2
            w2 = st["norm2.weight"]
            gates.append(read_rows(fused[:width], w2, C.cpu(), rho[l]))
            ups.append(read_rows(fused[width:], w2, C.cpu(), rho[l]))
            downs.append(write_cols(st["ffn.down.weight"], C.cpu(), alpha))
            origin += [(l, c) for c in range(width)]
        G = torch.cat(gates).to(self.device)
        U = torch.cat(ups).to(self.device)
        D = torch.cat(downs, dim=1)
        dnorm = D.norm(dim=0).to(self.device)
        total = int(G.shape[0])
        cand_params = 3 * sum(int(recs[l].state["ffn.down.weight"].numel()) for l in pool)
        # stage 1: write-energy prefilter over every pooled channel (fit tokens)
        energy = torch.zeros(total, dtype=torch.float64, device=self.device)
        fit_mask = self.calib.mask(self.calib.fit_rows)
        chunk = 2048
        for rows, v in self._v2_batches():
            m = fit_mask[rows].to(self.device)
            if not bool(m.any()):
                continue
            x = rms_normalize(z[rows], self.eps)[m]
            for s in range(0, x.shape[0], chunk):
                xs = x[s:s + chunk]
                a = F.silu(xs @ G.T) * (xs @ U.T)
                energy += a.pow(2).sum(dim=0).double()
        energy = energy * dnorm.double().pow(2)
        pool_size = min(total, max(kk, self.opt.prefilter_factor * kk))
        pre = torch.argsort(energy, descending=True)[:pool_size]
        pre = torch.sort(pre).values
        # stage 2: c14 pivoted Cholesky on the write-weighted activation Gram of the survivors
        gram = torch.zeros(pool_size, pool_size, dtype=torch.float64, device=self.device)
        Gp, Up, wp = G[pre], U[pre], dnorm[pre]
        for rows, v in self._v2_batches():
            m = fit_mask[rows].to(self.device)
            if not bool(m.any()):
                continue
            x = rms_normalize(z[rows], self.eps)[m]
            for s in range(0, x.shape[0], chunk):
                xs = x[s:s + chunk]
                a = F.silu(xs @ Gp.T) * (xs @ Up.T) * wp
                gram += (a.T @ a).double()
        picked = pivoted_cholesky_select(gram, kk)
        del gram
        if picked.numel() < kk:
            # rank-starved pool: top up by energy among the unpicked survivors (recorded)
            rest = [i for i in torch.argsort(energy[pre].cpu(), descending=True).tolist() if i not in set(picked.tolist())]
            picked = torch.sort(torch.cat([picked, torch.tensor(rest[: kk - picked.numel()], dtype=torch.long)])).values
        chosen = pre.cpu()[picked]
        with torch.no_grad():
            ffn.up.weight[:kk] = G[chosen.to(self.device)] / sbar
            ffn.up.weight[kk:] = U[chosen.to(self.device)] / sbar
            block.norm2.weight.fill_(sbar)
        prior = D[:, chosen]
        del G, U
        # the v2 FFN's own hidden rows -> ridge of `down` against the pooled source FFN writes
        captured: list[torch.Tensor] = []
        hook = ffn.down.register_forward_hook(lambda mod, inp, out: captured.append(inp[0]))
        pack = GramPack(kk, self.d, self.device)
        try:
            for rows, v in self._v2_batches():
                captured.clear()
                with torch.no_grad():
                    block.feed_forward_output(z[rows], v)
                hidden = captured[0]
                if block_end:
                    target = self._group_end[rows].to(self.device) - z[rows]
                else:
                    target = sum(recs[l].ffn[rows] for l in pool).to(self.device)
                for local, split in enumerate(self._row_splits(rows)):
                    if split is None:
                        continue
                    m = v[local]
                    pack.add(split, hidden[local][m].float(), target[local][m].float())
        finally:
            hook.remove()
        weight, ridge = pack.solve(prior)
        with torch.no_grad():
            ffn.down.weight.copy_(weight.to(self.device))
        self.param_methods.update({f"blocks.{k}.norm2.weight": "chart_scalar",
                                   f"blocks.{k}.ffn.up.weight": "union_coreset",
                                   f"blocks.{k}.ffn.down.weight": "ridge"})
        per_source = {}
        for idx in chosen.tolist():
            l = origin[idx][0]
            per_source[l] = per_source.get(l, 0) + 1
        installed = numel(ffn.up.weight.shape) + numel(ffn.down.weight.shape)
        self._component(k, "ffn", "union_coreset", cand_params, installed, selection=True,
                        diagnostics={"rho": {str(l): v for l, v in rho.items()}, "norm_scalar": sbar,
                                     "candidates": total, "prefilter_pool": pool_size, "kept": kk,
                                     "stage2_pivots": int(picked.numel()), "down_ridge": ridge,
                                     "target": ("alpha h_end C - z (F-BE block-end)" if block_end
                                                else "sum of the pooled source FFN writes"),
                                     "energy_kept_fraction": float(energy[chosen.to(self.device)].sum() / energy.sum().clamp_min(1e-300))},
                        sources=[{"source_layer": l, "channels": per_source.get(l, 0)} for l in pool])
        return {"rho": rho, "norm_scalar": sbar, "kept_per_source": per_source, "down_ridge": ridge}

    # ---------------------------------------------------------------- driver
    @torch.no_grad()
    def run(self) -> tuple[ChimeraOmniModel, dict[str, Any]]:
        t_start = time.time()
        self._retained: dict[int, list[float]] = {}
        if self.opt.chart is None:
            chart, chart_record = self.pass1_chart()
            if self.opt.save_chart:
                chart_record["saved"] = self.opt.save_chart
                chart_record["saved_sha256"] = c20.save_chart(self.opt.save_chart, chart, chart_record)
                log(f"chart saved to {self.opt.save_chart}")
        else:
            chart, chart_record = self.opt.chart.float(), dict(self.opt.chart_record or {})
        if tuple(chart.shape) != (self.src.hidden, self.d):
            raise SystemExit(f"chart is {tuple(chart.shape)}, expected {(self.src.hidden, self.d)}")
        self.chart, self.chart_record = chart, chart_record
        C = chart.to(self.device)

        with torch.device(self.device):
            model = ChimeraOmniModel(self.cfg)
        model.eval()
        use_chunked_gdn_scan(model, self.opt.gdn_chunk)
        # nothing random survives: every parameter starts at zero and is then folded or declared
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        alpha = self.fold_vocabulary(model, C)
        self._init_constants(model)

        h = self._source_embed()
        z = (model.embed.weight[self.calib.input_ids.to(self.device)].float()
             * self.calib.valid.unsqueeze(-1).to(self.device))
        table = self.alignment["trunk_table"]
        n_periods = int(self.cfg.num_layers) // self.src.period
        for p in range(n_periods):
            rows = [r for r in table if r["target_period"] == p]
            t0 = time.time()
            recs = self.collect_group(rows[0]["source_periods"], h, C, alpha)
            t_src = time.time() - t0
            for row in rows:
                k = int(row["target_layer"])
                block = model.blocks[k]
                pool = list(row["source_layers"])
                t1 = time.time()
                mix = self.fold_mixer(k, block, pool, recs, z, C, alpha)
                with torch.no_grad():
                    for r_, v in self._v2_batches():
                        z[r_] = z[r_] + block.attention_output(z[r_], token_valid_mask=v)
                block_end = self.opt.target_mode == "block_end" and int(row["slot"]) == self.src.period - 1
                ffn = self.fold_ffn(k, block, pool, recs, z, C, alpha, block_end=block_end)
                with torch.no_grad():
                    for r_, v in self._v2_batches():
                        z[r_] = z[r_] + block.feed_forward_output(z[r_], v)
                record = {"target_layer": k, "kind": block.kind, "source_layers": pool,
                          "mixer": {kk_: mix[kk_] for kk_ in ("norm_scalar", "rho")} | {
                              "out_held_rel_l2": mix["out_ridge"]["held_rel_l2"],
                              "out_prior_held_rel_l2": mix["out_ridge"]["prior_held_rel_l2"],
                              "heads": [(c["source_layer"], c["head"]) for c in mix["selection"]]},
                          "ffn": {"norm_scalar": ffn["norm_scalar"], "kept_per_source": ffn["kept_per_source"],
                                  "down_held_rel_l2": ffn["down_ridge"]["held_rel_l2"],
                                  "down_prior_held_rel_l2": ffn["down_ridge"]["prior_held_rel_l2"]},
                          "seconds": time.time() - t1}
                self.layers.append(record)
                log(f"layer {k} ({block.kind}) <- source {pool}: out held relL2 {mix['out_ridge']['held_rel_l2']}, "
                    f"down held relL2 {ffn['down_ridge']['held_rel_l2']} ({record['seconds']:.0f}s)")
            # period-end stream divergence on held rows (the dual-stream diagnostic)
            held = self.calib.mask(self.calib.held_rows).to(self.device)
            target = self._group_end.to(self.device)[held]
            got = z[held]
            self.layers[-1]["stream_divergence_held"] = {
                "rel_l2": float((got - target).norm() / target.norm().clamp_min(1e-30)),
                "cos_mean": float(F.cosine_similarity(got, target, dim=-1).mean()),
                "source_layers_through": (rows[0]["source_periods"][-1] + 1) * self.src.period - 1,
                "source_seconds": t_src}
            log(f"period {p}: stream held relL2 {self.layers[-1]['stream_divergence_held']['rel_l2']:.4f} "
                f"cos {self.layers[-1]['stream_divergence_held']['cos_mean']:.4f}")
            del recs
        # source periods past the last aligned period still run (the final stream is the target of the head)
        last_period = self.alignment["trunk_table"][-1]["source_periods"][-1]
        if last_period + 1 < self.src.num_layers // self.src.period:
            raise SystemExit("alignment left trailing source periods unfolded")

        # final norm scalar: the logit read of the source stream, charted
        fit = self.calib.mask(self.calib.fit_rows).to(self.device)
        s_final = float((rms(z, self.eps)[fit] / (alpha * rms(h, self.src.rms_eps)[fit]).clamp_min(1e-30)).mean())
        self.final_norm_record = self._final_norm(model, z, h, C, s_final)
        self.source_reference = self._source_reference(h)
        self.assembly = self._assembly_check(model, z)
        receipt = self._receipt(model, alpha, s_final)
        receipt["seconds_total"] = time.time() - t_start
        return model, receipt

    def _final_norm(self, model: ChimeraOmniModel, z: torch.Tensor, h: torch.Tensor, C: torch.Tensor,
                    s_final: float) -> dict[str, Any]:
        """The logit read: (z / rms(z)) . g must match (h / rms(h)) C, since W = H diag(w_final) C.

        scalar: g = s_final (plan 4.5).  diag (arm F-FND): g_j = sum_t u_tj y_tj / sum_t u_tj^2 over fit
        tokens, the per-channel least-squares solution (a diagonal ridge, closed form).  Held relL2 of
        both is recorded either way.
        """

        d = self.d
        num = torch.zeros(d, dtype=torch.float64, device=self.device)
        den = torch.zeros(d, dtype=torch.float64, device=self.device)
        held_terms = {"scalar": 0.0, "diag": 0.0, "ref": 0.0}
        held_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        for rows, v in self._v2_batches():
            for local, split in enumerate(self._row_splits(rows)):
                if split is None:
                    continue
                m = v[local]
                u = rms_normalize(z[rows][local][m].float(), self.eps)
                y = rms_normalize(h[rows][local][m].float(), self.src.rms_eps) @ C
                if split == "held":
                    held_rows.append((u, y))
                else:
                    num += (u * y).sum(dim=0).double()
                    den += (u * u).sum(dim=0).double()
        diag = (num / den.clamp_min(1e-30)).float()
        for u, y in held_rows:
            held_terms["scalar"] += float((u * s_final - y).pow(2).sum())
            held_terms["diag"] += float((u * diag - y).pow(2).sum())
            held_terms["ref"] += float(y.pow(2).sum())
        ref = max(held_terms["ref"], 1e-30)
        with torch.no_grad():
            if self.opt.final_norm == "diag":
                model.norm.weight.copy_(diag)
            elif self.opt.final_norm == "scalar":
                model.norm.weight.fill_(s_final)
            else:
                raise SystemExit(f"unknown final_norm {self.opt.final_norm!r}")
        self.param_methods["norm.weight"] = "chart_scalar" if self.opt.final_norm == "scalar" else "ridge"
        return {"mode": self.opt.final_norm, "scalar": s_final,
                "diag_quartiles": [float(q) for q in torch.quantile(diag.cpu(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))],
                "held_rel_l2_scalar": math.sqrt(held_terms["scalar"] / ref),
                "held_rel_l2_diag": math.sqrt(held_terms["diag"] / ref)}

    def _init_constants(self, model: ChimeraOmniModel) -> None:
        """AttnRes zero init (function-preserving) and the declared placeholders of what this build does not fold."""

        with torch.no_grad():
            for name, p in model.named_parameters():
                if "attention_residual" in name or "feed_forward_residual" in name:
                    if name.endswith("norm.weight"):
                        p.fill_(1.0)
                        self.param_methods[name] = "unit"
                    else:
                        self.param_methods[name] = "zero_init_attnres"   # query 0, prefix bias 0 (attnres_init="zero")
                elif name.startswith(("native_media_bridge", "native_media_head")):
                    if name.endswith("norm.weight"):
                        p.fill_(1.0)
                        self.param_methods[name] = "unit"
                    else:
                        self.param_methods[name] = "zero_placeholder"
        self._component(None, "native_media_bridge+head", "zero_placeholder", 0,
                        sum(p.numel() for n, p in model.named_parameters()
                            if n.startswith(("native_media_bridge", "native_media_head"))),
                        diagnostics={"not_built": "raw boundary and x0 head ridge fits (plan 4.5); zero, not random"})

    def _source_reference(self, h: torch.Tensor) -> dict[str, Any]:
        """The source's own held-row NLL from the final source stream (a reference column, never a gate)."""

        w_final = self.src.final_norm().to(self.device)
        head = self.src.head()
        nll, count = 0.0, 0
        per_row = {}
        for r in self.calib.held_rows:
            v = self.calib.valid[r]
            length = int(v.sum())
            if length < 2:
                continue
            x = rms_normalize(h[r, :length - 1].float(), self.src.rms_eps) * w_final
            labels = self.calib.input_ids[r, 1:length].to(self.device)
            logz = torch.full((length - 1,), -float("inf"), device=self.device)
            picked = torch.zeros(length - 1, device=self.device)
            for start in range(0, head.shape[0], 32768):
                rows = slice(start, min(head.shape[0], start + 32768))
                logits = x @ head[rows].to(self.device).float().T
                logz = torch.logaddexp(logz, torch.logsumexp(logits, dim=-1))
                inside = (labels >= rows.start) & (labels < rows.stop)
                if bool(inside.any()):
                    idx = torch.nonzero(inside).flatten()
                    picked[idx] = logits[idx, labels[idx] - rows.start]
            row_nll = float((logz - picked).sum())
            per_row[str(r)] = row_nll / (length - 1)
            nll += row_nll
            count += length - 1
        return {"held_nll_per_token": nll / max(count, 1), "held_tokens_scored": count, "per_row": per_row,
                "what": "the source trunk's own next-token NLL on the held rows (reference, not a gate)"}

    def _assembly_check(self, model: ChimeraOmniModel, z: torch.Tensor) -> dict[str, Any]:
        """The assembled model's real trunk (AttnRes and all) must reproduce the fold's own stream direction."""

        r = self.calib.held_rows[0]
        v = self.calib.valid[r:r + 1].to(self.device)
        ids = self.calib.input_ids[r:r + 1].to(self.device)
        use_chunked_gdn_scan(model, self.opt.gdn_chunk)
        with torch.no_grad():
            x = model.embed(ids)
            out, _, _ = model._forward_trunk(x, v)
            got = rms_normalize(out[v].float(), self.eps)
            want = rms_normalize(z[r:r + 1][v].float(), self.eps)
        rel = float((got - want).norm() / want.norm().clamp_min(1e-30))
        if not math.isfinite(rel) or rel > ASSEMBLY_TOLERANCE:
            raise SystemExit(f"assembled model disagrees with the fold's stream: relL2 {rel:.3e} > {ASSEMBLY_TOLERANCE}")
        return {"held_row": r, "normalised_stream_rel_l2": rel, "tolerance": ASSEMBLY_TOLERANCE}

    def _receipt(self, model: ChimeraOmniModel, alpha: float, s_final: float) -> dict[str, Any]:
        unnamed = [n for n, _ in model.named_parameters() if n not in self.param_methods]
        if unnamed:
            raise SystemExit(f"{len(unnamed)} parameters have no fold method: {unnamed[:8]}")
        nonfinite = [n for n, p in model.named_parameters() if not bool(torch.isfinite(p).all())]
        if nonfinite:
            raise SystemExit(f"non-finite folded tensors: {nonfinite[:8]}")
        below = [c for c in self.components if c["selection_component"] and not c["clears_4x"]]
        if below and self.opt.require_4x:
            raise SystemExit(f"selection components below 4x: {[(c['target_layer'], c['component'], c['compression']) for c in below]}")
        methods: dict[str, int] = {}
        for name, p in model.named_parameters():
            methods[self.param_methods[name]] = methods.get(self.param_methods[name], 0) + p.numel()
        retained = {str(l): (1.0 - e / t) if t > 0 else None for l, (e, t) in sorted(self._retained.items())}
        return {
            "schema": RECEIPT_SCHEMA,
            "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "path": os.path.abspath(__file__),
                     "sha256": TOOL_SHA256, "c14_sha256": DEP_SHA256["c14"],
                     "c19_sha256": DEP_SHA256["c19"], "c20_sha256": DEP_SHA256["c20"],
                     "sha256_when": "hashed at import (the code that ran)",
                     "chimera_src": CHIMERA_SRC},
            "ruling": "cyril-v2-arch-and-fold-init-2026-10-21",
            "source": {"label": self.src.label, "path": self.src.path, "hidden": self.src.hidden,
                       "layers": self.src.num_layers, "period": self.src.period, "tied": self.src.tied,
                       "import": "chimera_omni.importing.qwen35.import_qwen35_state per period (exact importer)",
                       "output_gate": "allow_unverified_output_gate=True (Q1 observation stamped in the period config)"},
            "target": {"profile": self.cfg.metadata.get("profile"), "config": self.cfg.to_manifest(),
                       "parameters": sum(p.numel() for p in model.parameters())},
            "calibration": self.calib.record,
            "alignment": self.alignment, "alignment_sha256": self.alignment_sha256,
            "chart": {**self.chart_record, "chart_sha256": c20.chart_sha256(self.chart)},
            "vocabulary": self.vocab_record,
            "target_mode": self.opt.target_mode,
            "stream_scale_alpha": alpha,
            "final_norm_scalar": s_final,
            "final_norm": self.final_norm_record,
            "retained_energy_held_per_source_layer": retained,
            "layers": self.layers,
            "components": self.components,
            "parameters_by_method": methods,
            "copy_methods": 0,
            "four_x_rule": "enforced" if self.opt.require_4x else "OFF (identity mechanics test only; not a fold)",
            "not_built": ["K3 form-E union term (needs the P1 K3 row refold on fit tokens; plan section 7 items 2-3)",
                          "media-unit channels and heads (plan 4.4 item 3)",
                          "raw boundary + x0 head ridge fits (plan 4.5): zero placeholders",
                          "arms F-C2, F-K3DIRECT, F-K3MIX/MLA (plan 10 step 12)"],
            "source_reference": self.source_reference,
            "assembly_check": self.assembly,
            "timings": self.timings,
            "policy": "every fit number is a diagnostic; refusals are structural (geometry, non-finite, unnamed "
                      "parameter, assembly mismatch, selection below 4x)",
        }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def save_checkpoint(model: ChimeraOmniModel, out: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    from chimera_omni.importing.safetensors_io import write_safetensors

    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}
    if model.cfg.tie_word_embeddings:
        state["lm_head.weight"] = state["embed.weight"]
    index = write_safetensors(state, out, config=model.cfg.to_manifest(),
                              source={"fold": TOOL_NAME, "tool_version": TOOL_VERSION,
                                      "source": receipt["source"]["label"], "ruling": receipt["ruling"]},
                              metadata={"tool": TOOL_NAME, "tool_version": TOOL_VERSION})
    files = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    return {"path": str(out), "files": files, "index_metadata": index.get("metadata", {})}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C21: fold a Qwen3.5/3.8 trunk into a v2 profile (closed form)")
    parser.add_argument("--source", required=True, help="HF checkpoint directory (T: the run-1 full-KD student)")
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--profile", required=True, choices=sorted(V2_PROFILES))
    parser.add_argument("--corpus", required=True, help="qwen_input_ids.safetensors (fit/held rows)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rows", type=int, default=None, help="smoke: first N corpus rows only")
    parser.add_argument("--seq-len", type=int, default=None, help="smoke: truncate rows")
    parser.add_argument("--prefilter-factor", type=int, default=4)
    parser.add_argument("--gdn-chunk", type=int, default=64)
    parser.add_argument("--tie", choices=("head", "joint"), default="head")
    parser.add_argument("--final-norm", choices=("scalar", "diag"), default="scalar",
                        help="diag = arm F-FND (per-channel closed-form logit read)")
    parser.add_argument("--target-mode", choices=("writes", "block_end"), default="writes",
                        help="block_end = arm F-BE (plan 4.6 b)")
    parser.add_argument("--chart", default=None, help="reuse a c20 chart (.safetensors); else pass 1 fits and saves one")
    parser.add_argument("--save-chart", default=None)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; a fold never overwrites")
    source = QwenTrunkSource.from_dir(args.source, label=args.source_label)
    cfg = V2_PROFILES[args.profile]()
    calib = load_corpus(args.corpus, rows=args.rows, seq_len=args.seq_len)
    chart = chart_record = None
    if args.chart:
        chart, chart_record = c20.load_chart(args.chart)
        chart_record["path"] = args.chart
    opt = FoldOptions(device=args.device, batch=args.batch, prefilter_factor=args.prefilter_factor,
                      gdn_chunk=args.gdn_chunk, tie=args.tie, target_mode=args.target_mode, final_norm=args.final_norm,
                      chart=chart, chart_record=chart_record, save_chart=args.save_chart if chart is None else None)
    log(f"source {source.label} ({source.num_layers} layers, {source.hidden}) -> {args.profile} "
        f"({cfg.num_layers} layers, {cfg.hidden_size}); corpus rows {calib.input_ids.shape[0]} x {calib.input_ids.shape[1]}")
    fold = Fold(source, cfg, calib, opt)
    model, receipt = fold.run()
    receipt["checkpoint"] = save_checkpoint(model, out / "child", receipt)
    receipt["argv"] = sys.argv
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n", encoding="utf-8")
    log(f"wrote {out / 'child'} and {out / 'receipt.json'} ({receipt['seconds_total']:.0f}s)")
    print(json.dumps({"out": str(out), "parameters": receipt["target"]["parameters"],
                      "assembly_rel_l2": receipt["assembly_check"]["normalised_stream_rel_l2"],
                      "source_held_nll": receipt["source_reference"]["held_nll_per_token"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
