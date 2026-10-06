#!/usr/bin/env python3
"""C10 -- the union install: K3 form-E FFNs lifted into the Chimera trunk's MLP rows.

Run-book row (COMPOSITE_STEP1_RUNBOOK.md section 3, S5).  Per trunk layer k the
child's fused ``blocks.k.ffn.up.weight`` becomes

    [ gate_q | gate_k3 ]        Q = the trunk's own (C2 coreset) rows, ALWAYS kept
    [  up_q  | s . up_k3 ]      k3 = ``--k3-channels`` rows chosen by a union coreset

and ``blocks.k.ffn.down.weight`` [h x (Q + k3)] is a prior-anchored ridge refit of
the WHOLE union against

    Y(x) = MLP_parent(x) + sum_{l in align(k)} FFN_k3,l ( T_k x + b_k ) U_k^T

on the trunk's own FFN-input rows x (post ``norm2``), where ``MLP_parent`` is the
pristine S0 parent block evaluated at the child's input (qf_mlp's target) and
``FFN_k3,l`` is the form-E K3 layer evaluated EXACTLY in K3 space (latent down,
kept SiTU channels, refit ``w2p``, ``routed_expert_norm`` -- a per-token RMSNorm,
kept as the nonlinearity it is -- ``routed_expert_up_proj``, plus both verbatim
shared experts; the dense K3 layer 0 is a plain SiTU MLP).

CHARTS -- c17's convention, verbatim (c17_chart_fit.py lines 9-10, 909):

    C_in  [d_k3 x d_trunk]   y_hat = x_k3    @ C_in  + C_in_bias    (K3 -> trunk)
    C_out [d_trunk x d_k3]   x_hat = y_trunk @ C_out + C_out_bias   (trunk -> K3)

so the trunk -> K3 map this tool applies is ``T = C_out^T`` [d_k3 x d_trunk] with
``b = C_out_bias``, and the K3 -> trunk map is ``U = C_in^T`` [d_trunk x d_k3] with
``C_in_bias``.  For the ridge arm (the chart of record) C_in and C_out are fitted
INDEPENDENTLY, so C_out != C_in^T and the direction matters; ``Chart`` stores the
two maps under their own names (``trunk_to_k3`` / ``k3_to_trunk``) and nothing in
this file ever transposes one into the other.  The chart files are what c17 writes:
``L{k:02d}/ridge.safetensors`` (``--chart-arm ridge``, default), ``svd_chart.safetensors``
or ``procrustes.safetensors``, beside ``layer.json``; ``alignment.json`` in the same
directory is c17's ``{schema, summary, table: [{trunk, k3_layers, ...}]}``.

Native boundary gate (v1.2): production now REQUIRES each arm's
``native_boundary_provenance`` (schema ``neural-forge.c17-native-chart-boundaries.v1``).
Its ``maps.C_out`` binds host.ffn_input -> k3.ffn_input; ``maps.C_in`` binds
k3.ffn_output -> host.ffn_output. Domain/codomain each carry family, boundary,
layer and capture_receipt {path, sha256}; lists pair pooled source layers.
Receipts and actual boundary tensors/positions/splits are checked by
``native_chart_boundaries``. An input-space chart is not evidence for a
write-space map. The old C17 fitter does NOT emit this contract: an authentic
K3 capture + paired fitter are still needed; no existing charts are relabeled
or refitted here. Geometry/algebra above is unchanged. This consistency check
is not an independent proof that declared capture code is trustworthy or that
the fitted maps transfer intelligence; actual source/held functional review remains mandatory.
There is currently no production-native chart admission route: load_chart refuses
before chart I/O until a reviewed exact-installed evaluator and acceptance policy
exist. Renaming candidates or asserting affirmative metadata cannot promote them.
native_chart_boundaries remains a provenance-only helper for candidate fitting.

Legacy mechanics only: ``--synthetic-fixture-root`` must name a marked unit-test
workspace with CPU-only hidden <=128, <=8-layer geometry and explicit unit-test
checkpoint source markers. Its receipt is labeled ``synthetic_fixture: true``.
Identity and extra-block chart routes have no production native-boundary contract
and are refused. No large-model/GPU override exists for the fixture route.

Biases.  The K3 REFERENCE (the targets) applies the input bias exactly.  A bias-free
``FeedForward`` row cannot carry the constant pre-activation shift ``w_c . b`` the
input bias adds, so the lifted rows drop it and the receipt records the dropped
shift's energy on the fit rows per source (``chart_bias.input``); the output bias
``C_in_bias`` is a constant vector per token, unrepresentable by the union FFN, so it
is EXCLUDED from Y and recorded with its energy fraction (``chart_bias.output``).
Either exceeding ``--chart-bias-ceiling`` is a named refusal.

The lift, form E -> trunk width, is the composition of linear maps and is exact:

    routed  gate rows  W1_s . W_down . T         [K x h]      (test 6: == form M's
            up rows    W3_s . W_down . T                       w1_m . T)
            down prior U . W_up . diag(g) . W2p / rms_mean   [h x K]
    shared  gate rows  Wsg . T, up rows Wsu . T, down prior U . Wsd
    dense   gate rows  G . T,   up rows U_k3 . T, down prior U . D

The routed read-out passes through ``routed_expert_norm``; the prior linearises it
with the MEAN token rms over the fit rows (a scalar; never a division by a norm
gain -- section 3b) and the ridge refit absorbs the rest.  ``g`` multiplies; an
exact-zero gain coordinate is counted in the receipt and never divided by.

SiTU -> SiLU.  The trunk is a SiLU-GLU (Qwen) and ``ChimeraConfig`` carries ONE
``activation_kind``; K3's rows are SiTU_{4,25}.  Every lifted K3 channel gets a
per-channel scalar ``s_c`` on its up row, the least-squares fit of
``silu(g.x) (s u.x)`` to ``situ(g.x, u.x)`` over the fit rows, and the receipt
records the relL2 that substitution costs (test 4).

Union coreset.  Features A = [ silu(gq.x) (uq.x) | silu(g.x)(s u.x) ] over the fit
rows; ONE fp64 Gram of the full pool on the host, accumulated in column blocks on
``--device``.  The Qwen columns are pinned; the K3 share is a pivoted Cholesky
(``c14.pivoted_columns_gram``) on the Schur complement of the K3 pool given the
Qwen block (formed in place, block by block, never as two extra pool-sized
matrices), so a K3 channel is chosen for what it adds to rows already in the
layer, not for what it duplicates.  The check-split Gram is accumulated in a
second, cheap pass over the check rows on the KEPT columns only, after the full
Gram is sliced and freed -- one pool-sized fp64 Gram is ever resident.  The refit
is ``c14.law_abiding_solve_gram`` (lambda ladder, internal check split from
``qf_corpus.fit_check_rows``, ``update_scale <= 1`` veto) with
``c14.prior_anchored_gram`` wrapped so a Cholesky refusal is raised as
``qf_solve.GramNotPositiveDefinite`` WITH the offending channels (section 3b:
Gram defects are named).

Veto semantics (run-book S5 gate): ``veto_binding`` per layer is true when THE
LAW's free solve wanted more than the prior (``ridge.veto_binding``) OR when the
K3 term carries no energy on the held rows (``k3_rows_function_absent``) -- rows in
name only.  The receipt says which, out loud, and counts the layers.

Every receipt number is measured on the weights that are SHIPPED: the solution is
cast to the trunk's FFN dtype first, then scored (``heldout_relL2``, energy,
``update_scale``); the fp32 ladder numbers stay under ``ridge`` and the cast delta
is recorded.

STREAMED CAPTURE (default ``--engine streamed``).  The residual stream lives on the
host in the model dtype, one row per evidence token; block k is built on
``--device`` from the trunk's tensor views (plus the already-installed union FFNs
for k' < k), applied batch by batch, and freed.  A group of ``--group-size``
layers is captured by running the group's blocks with their TRUNK FFNs from the
group boundary (the FFN input of every block in the group is taken from a
forward pre-hook on ``block.ffn``), the group is solved and installed, and the
residual is then advanced through the group's NEW blocks -- qf_mlp's ordering
law, exactly as the resident model would see it (``--engine resident``, test
only, loads the whole model and is proven byte-identical on the synthetic
fixture).  Two streamed passes over the trunk in total, ~1 block of device
residency, the 4096 x vocab logits never formed.  Requires ``residual_mode ==
"none"`` (the S0 / S3 trunks are).

K3 layers are loaded lazily -- only ``alignment[k]`` for the trunk layer being
solved -- moved to the device once per install, and released.

PERSISTENCE.  Every solved layer is written to ``<out>.partial/L{k:02d}.safetensors``
(``up`` / ``down`` in the FFN dtype) with ``L{k:02d}.json`` (the layer record) the
moment it completes; ``partial_inputs.json`` seals the inputs' identity.
``--resume`` re-derives that identity (trunk / parent / evidence / chart digests,
the K3 files each completed layer consumed), reloads the completed layers into the
engine in order and continues from the first missing one; ``--stop-after-layer N``
returns exit code 3 after persisting layer N (detached runs under the SSH cap).
``--overwrite`` only ever removes a directory that carries this tool's marker or
receipt, and refuses when ``--out`` overlaps ANY input path.

Size: the child must have a header count STRICTLY below the parent record's
``parameter_count`` (``--parent-parameter-count`` / ``--parent-record``; default the
S0 parent trunk's own headers).  Refused up front from the arithmetic and again
after the write from ``oc_seal.safetensors_tensor_counts`` (test 5); a refused
child is removed.

Outputs
    <out>/model-*.safetensors + model.safetensors.index.json + config.json
        (``safetensors_io.write_safetensors``; config = the trunk's manifest with
        ``intermediate_size`` = ``intermediate_sizes[*]`` = Q + k3, uniform, so the
        qwen35 export's one-width rule holds)
    <out>/union_receipt.json  (+ ``--receipt`` copy) per layer: relL2 held-out of the
        union FFN vs Y, K3-term energy fraction, veto_binding, update_scale, ...

``--k3-channels 0`` copies the trunk byte-for-byte (test 1) and records that.

S5 launch (run book, with the K3 config named so the receipt carries its sha):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 $PY c10_union_install.py \\
      --trunk composite/s3/c2k4352-chimera-bf16 --parent-trunk composite/s0/q38-27bu-chimera-bf16-sep04 \\
      --k3-dense composite/k3/dense --k3-form E --k3-config /mnt/lexar/models/moonshotai/Kimi-K3/config.json \\
      --charts composite/charts --chart-arm ridge --evidence composite/corpus/omni-evidence-27b-v1.json \\
      --k3-channels 12288 --device cuda:0 --out /mnt/lexar/omni-campaign/composite/oc02/children/OC02-F-Q38-27BU-C2K4352K3U \\
      --receipt composite/s5/union_receipt.json [--resume]

Tests: test_c10_union_install.py (CPU, synthetic Chimera + synthetic form-E K3).
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

TOOL_NAME = "c10_union_install"
TOOL_VERSION = "1.2.0"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOCAL_CHIMERA_SRC = "C:/Users/cereal/Documents/ai training/chimera-omni/src"


def _resolve_dir(env: str, candidates: list[str], probes: list[str], label: str) -> str:
    override = os.environ.get(env)
    options = ([override] if override else []) + candidates
    for cand in options:
        if cand and all(os.path.exists(os.path.join(cand, p)) for p in probes):
            return os.path.abspath(cand)
    raise SystemExit(f"{label} ({', '.join(probes)}) was not found in {options}; set {env}")


K3_LANE_DIR = _resolve_dir("K3_LANE_DIR", [os.path.join(ROOT, "k3", "harness"), os.path.join(ROOT, "k3")],
                           ["k3_common.py", "c14_union_coreset.py"], "the k3 lane")
QF_LANE_DIR = _resolve_dir("QF_LANE_DIR", [os.path.join(ROOT, "qwen-fold")],
                           ["qf_solve.py", "qf_corpus.py"], "the qwen-fold lane")
LEDGER_DIR = _resolve_dir("OC_LEDGER_DIR", [os.path.join(ROOT, "ledger")], ["oc_seal.py"], "the ledger")
CHIMERA_SRC = _resolve_dir("CHIMERA_SRC", [os.path.join(ROOT, "import", "rq27b", "tree_c5e6052", "src"),
                                           LOCAL_CHIMERA_SRC],
                           [os.path.join("chimera_omni", "config.py")], "the chimera-omni source tree")
for _dir in (K3_LANE_DIR, QF_LANE_DIR, LEDGER_DIR, CHIMERA_SRC):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

import c14_union_coreset as c14  # noqa: E402
import oc_seal  # noqa: E402
import qf_corpus  # noqa: E402
import qf_solve  # noqa: E402
from k3_common import (  # noqa: E402
    canonical_device,
    peak_rss_bytes,
    rms_norm,
    sha256_file,
    situ,
    write_receipt,
    write_scores,
)
from chimera_omni.config import ChimeraConfig  # noqa: E402
from chimera_omni.importing.safetensors_io import (  # noqa: E402
    CONFIG_FILENAME,
    INDEX_FILENAME,
    load_chimera_checkpoint,
    read_checkpoint_config,
    read_safetensors,
    shard_paths,
    write_safetensors,
)
from chimera_omni.modeling.layers import ChimeraBlock, FeedForward  # noqa: E402

LAMBDA_LADDER = c14.LAMBDA_LADDER
RIDGE_REL = c14.RIDGE_REL
K3_FORM_E_KEYS = ("w1_s", "w3_s", "w2p",
                  "block_sparse_moe.routed_expert_down_proj.weight",
                  "block_sparse_moe.routed_expert_up_proj.weight",
                  "block_sparse_moe.routed_expert_norm.weight",
                  "block_sparse_moe.shared_experts.gate_proj.weight",
                  "block_sparse_moe.shared_experts.up_proj.weight",
                  "block_sparse_moe.shared_experts.down_proj.weight")
K3_DENSE_KEYS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
#: below this held-out energy fraction the K3 rows carry no function -- the veto
#: binds by definition, whatever the ridge said (run-book S5: rows in name only)
K3_ENERGY_FLOOR = 1e-6
NONFINITE_CEILING = 0.02
#: c17's chart files per arm (c17_chart_fit.py:901-903) and its recorded convention
CHART_ARM_FILES = {"ridge": "ridge.safetensors", "svd": "svd_chart.safetensors", "procrustes": "procrustes.safetensors"}
C17_CONVENTION = "y_hat = x_k3 @ C_in + C_in_bias; x_hat = y_trunk @ C_out + C_out_bias"
C17_ALIGNMENT_SCHEMA = "chimera_omni.composite.c17_alignment.v1"
NATIVE_CHART_SCHEMA = "neural-forge.c17-native-chart-boundaries.v1"
FIXTURE_MARKER = "c10-synthetic-fixture.json"
PARTIAL_SCHEMA = "neural-forge-c10-partial-v1"
PARTIAL_INPUTS = "partial_inputs.json"
OUT_MARKER = "c10_union_install.marker.json"
EXIT_STOPPED = 3


def log(message: str) -> None:
    print(f"[{TOOL_NAME}] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# section 3b: Gram defects are NAMED.  c14's ridge raises the bare LAPACK line;
# wrap it once so every ladder rung of the union refit refuses with the census.
# --------------------------------------------------------------------------
_c14_prior_anchored_gram = c14.prior_anchored_gram


def named_prior_anchored_gram(gram: torch.Tensor, cross: torch.Tensor, prior: torch.Tensor,
                              rel: float = RIDGE_REL) -> torch.Tensor:
    if not bool(torch.isfinite(gram).all()):
        lam = rel * float(torch.diagonal(gram).mean())
        report = qf_solve.gram_defect_report(gram, lam, rel)
        raise qf_solve.GramNotPositiveDefinite(qf_solve._defect_message(report, "c10 union refit"), report)
    try:
        return _c14_prior_anchored_gram(gram, cross, prior, rel=rel)
    except qf_solve.GramNotPositiveDefinite:
        raise
    except Exception as error:  # torch._C._LinAlgError and whatever LAPACK adds
        lam = rel * float(torch.diagonal(gram).mean())
        report = qf_solve.gram_defect_report(gram, lam, rel)
        report["cholesky_error"] = f"{type(error).__name__}: {error}"
        raise qf_solve.GramNotPositiveDefinite(
            qf_solve._defect_message(report, "c10 union refit"), report) from error


c14.prior_anchored_gram = named_prior_anchored_gram


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().to(torch.float32).cpu().contiguous().numpy()
    return sha256_bytes(arr.tobytes())


def sha256_tensor_bytes(tensor: torch.Tensor) -> str:
    """The tensor's own bytes in its own dtype (what the shipped file carries)."""
    t = tensor.detach().cpu().contiguous()
    return sha256_bytes(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes())


def glu(kind: str, gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """The gated activation a source computes: SiLU-GLU or SiTU, fp32 in / fp32 out."""
    if kind == "silu":
        return F.silu(gate) * up
    if kind == "situ":
        if math.isinf(beta) and (linear_beta is None or math.isinf(linear_beta)):
            return F.silu(gate) * up
        return situ(gate, up, beta, None if (linear_beta is None or math.isinf(linear_beta)) else linear_beta)
    raise SystemExit(f"unknown activation kind {kind!r}")


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    denom = float(target.norm())
    if denom == 0.0:
        return 0.0 if float(pred.norm()) == 0.0 else float("inf")
    return float((pred - target).norm() / denom)


def zero_gain_census(name: str, weight: torch.Tensor) -> dict[str, Any]:
    zero = torch.nonzero(weight.detach().float().cpu() == 0).flatten()
    return {"tensor": name, "zero_gain_count": int(zero.numel()), "zero_gain_coordinates_first": zero[:16].tolist(),
            "rule": "multiplied, never divided by (run-book section 3b)"}


def checkpoint_files(root: Path) -> list[Path]:
    files = [Path(p) for p in shard_paths(root)]
    for name in (INDEX_FILENAME, CONFIG_FILENAME):
        if (root / name).is_file():
            files.append(root / name)
    return files


def directory_file_digests(root: Path) -> dict[str, str]:
    return {p.name: sha256_file(str(p)) for p in checkpoint_files(root)}


def rss_gigabytes() -> float | None:
    peak = peak_rss_bytes()
    return round(peak / 1e9, 3) if peak else None


# --------------------------------------------------------------------------
# K3 form-E layers
# --------------------------------------------------------------------------
@dataclass
class K3Config:
    rms_norm_eps: float = 1e-5
    beta: float = 4.0
    linear_beta: float | None = 25.0


def load_k3_config(path: str | None, args: argparse.Namespace) -> tuple[K3Config, dict[str, Any]]:
    cfg = K3Config(rms_norm_eps=float(args.k3_rms_eps), beta=float(args.k3_situ_beta),
                   linear_beta=None if args.k3_situ_linear_beta <= 0 else float(args.k3_situ_linear_beta))
    record: dict[str, Any] = {"source": "flags (--k3-config-from-flags)"}
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        text = raw.get("text_config", raw)
        if "rms_norm_eps" in text:
            cfg.rms_norm_eps = float(text["rms_norm_eps"])
        if text.get("activation_situ_beta") is not None:
            cfg.beta = float(text["activation_situ_beta"])
        if "activation_situ_linear_beta" in text:
            cfg.linear_beta = None if text["activation_situ_linear_beta"] is None else float(
                text["activation_situ_linear_beta"])
        record = {"source": os.path.abspath(path), "sha256": sha256_file(path)}
    elif not getattr(args, "k3_config_from_flags", False):
        raise SystemExit("--k3-config <K3 config.json> is required so the receipt carries the constants' sha256 "
                         "(/mnt/lexar/models/moonshotai/Kimi-K3/config.json on the box); tests may pass "
                         "--k3-config-from-flags to use --k3-rms-eps / --k3-situ-beta / --k3-situ-linear-beta")
    record.update({"rms_norm_eps": cfg.rms_norm_eps, "activation_situ_beta": cfg.beta,
                   "activation_situ_linear_beta": cfg.linear_beta})
    return cfg, record


def _check_finite(path: str, tensors: dict[str, torch.Tensor]) -> None:
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor.float()).all()):
            raise SystemExit(f"{path}: tensor {name!r} carries non-finite values -- refused before any Gram")


@dataclass
class K3Layer:
    """One K3 layer as S2 wrote it: form E (MoE) or the verbatim dense MLP (layer 0)."""

    index: int
    path: str
    sha256: str
    kind: str                      # "form_E" | "dense"
    tensors: dict[str, torch.Tensor]
    cfg: K3Config

    @property
    def hidden(self) -> int:
        if self.kind == "dense":
            return int(self.tensors["mlp.gate_proj.weight"].shape[1])
        return int(self.tensors["block_sparse_moe.routed_expert_down_proj.weight"].shape[1])

    @property
    def routed_width(self) -> int:
        return 0 if self.kind == "dense" else int(self.tensors["w1_s"].shape[0])

    @property
    def shared_width(self) -> int:
        return 0 if self.kind == "dense" else int(self.tensors["block_sparse_moe.shared_experts.gate_proj.weight"].shape[0])

    @property
    def dense_width(self) -> int:
        return int(self.tensors["mlp.gate_proj.weight"].shape[0]) if self.kind == "dense" else 0

    def device_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """ONE fp32 copy of the layer on ``device`` (minor finding: never per chunk)."""
        keys = K3_DENSE_KEYS if self.kind == "dense" else K3_FORM_E_KEYS
        return {k: self.tensors[k].to(device=device, dtype=torch.float32) for k in keys}

    def zero_gains(self) -> dict[str, Any] | None:
        if self.kind != "form_E":
            return None
        return zero_gain_census(f"K3 L{self.index:02d} routed_expert_norm",
                                self.tensors["block_sparse_moe.routed_expert_norm.weight"])

    def reference(self, xk: torch.Tensor, dev: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The K3 FFN evaluated EXACTLY in K3 space on rows ``xk`` [n x H] (fp32).

        ``dev`` is ``device_tensors(xk.device)``.  Returns (output [n x H], routed
        pre-norm accumulator y_lat [n x d] or None).  The routed path keeps
        ``routed_expert_norm`` as the per-token RMSNorm it is.
        """
        c = self.cfg
        if self.kind == "dense":
            out = glu("situ", xk @ dev["mlp.gate_proj.weight"].T, xk @ dev["mlp.up_proj.weight"].T,
                      c.beta, c.linear_beta) @ dev["mlp.down_proj.weight"].T
            return out, None
        latent = xk @ dev["block_sparse_moe.routed_expert_down_proj.weight"].T
        psi = glu("situ", latent @ dev["w1_s"].T, latent @ dev["w3_s"].T, c.beta, c.linear_beta)
        y_lat = psi @ dev["w2p"].T
        normed = rms_norm(y_lat, dev["block_sparse_moe.routed_expert_norm.weight"], c.rms_norm_eps)
        routed = normed @ dev["block_sparse_moe.routed_expert_up_proj.weight"].T
        shared = glu("situ", xk @ dev["block_sparse_moe.shared_experts.gate_proj.weight"].T,
                     xk @ dev["block_sparse_moe.shared_experts.up_proj.weight"].T,
                     c.beta, c.linear_beta) @ dev["block_sparse_moe.shared_experts.down_proj.weight"].T
        return routed + shared, y_lat

    def release(self) -> None:
        """Drop the host tensors (lazy loading: a layer lives for one trunk layer)."""
        self.tensors = {}


def k3_layer_files(root: str, index: int) -> tuple[str, str]:
    layer_dir = os.path.join(root, f"L{index:02d}")
    ffn = os.path.join(layer_dir, "ffn.safetensors")
    mlp = os.path.join(layer_dir, "mlp.safetensors")
    if os.path.isfile(ffn):
        return ffn, "form_E"
    if os.path.isfile(mlp):
        return mlp, "dense"
    raise SystemExit(f"K3 layer {index}: neither {ffn} nor {mlp} exists")


def k3_layer_header(root: str, index: int) -> dict[str, Any]:
    """Kind / hidden / widths from the safetensors HEADER only (no tensor bytes)."""
    from safetensors import safe_open

    path, kind = k3_layer_files(root, index)
    with safe_open(path, "pt") as handle:
        keys = set(handle.keys())
        needed = K3_FORM_E_KEYS if kind == "form_E" else K3_DENSE_KEYS
        missing = [k for k in needed if k not in keys]
        if missing:
            raise SystemExit(f"{path} is not a {'form-E layer' if kind == 'form_E' else 'dense K3 MLP'}: missing {missing}")
        shape = {k: tuple(handle.get_slice(k).get_shape()) for k in needed}
    if kind == "dense":
        return {"index": index, "path": path, "kind": kind, "hidden": int(shape["mlp.gate_proj.weight"][1]),
                "routed_width": 0, "shared_width": 0, "dense_width": int(shape["mlp.gate_proj.weight"][0])}
    return {"index": index, "path": path, "kind": kind,
            "hidden": int(shape["block_sparse_moe.routed_expert_down_proj.weight"][1]),
            "routed_width": int(shape["w1_s"][0]),
            "shared_width": int(shape["block_sparse_moe.shared_experts.gate_proj.weight"][0]), "dense_width": 0}


def load_k3_layer(root: str, index: int, cfg: K3Config) -> K3Layer:
    from safetensors.torch import load_file

    path, kind = k3_layer_files(root, index)
    tensors = load_file(path)
    needed = K3_FORM_E_KEYS if kind == "form_E" else K3_DENSE_KEYS
    missing = [k for k in needed if k not in tensors]
    if missing:
        raise SystemExit(f"{path} is not a {'form-E layer' if kind == 'form_E' else 'dense K3 MLP'}: missing {missing}")
    _check_finite(path, tensors)
    return K3Layer(index, path, sha256_file(path), kind, tensors, cfg)


def discover_k3_layers(root: str) -> list[int]:
    found = []
    for entry in sorted(os.listdir(root)):
        if entry.startswith("L") and entry[1:].isdigit() and os.path.isdir(os.path.join(root, entry)):
            if os.path.isfile(os.path.join(root, entry, "ffn.safetensors")) or os.path.isfile(
                    os.path.join(root, entry, "mlp.safetensors")):
                found.append(int(entry[1:]))
    if not found:
        raise SystemExit(f"{root} holds no L*/ffn.safetensors or L*/mlp.safetensors")
    return found


def form_m_from_form_e(layer: K3Layer) -> dict[str, torch.Tensor]:
    """``c14.build_form_m`` in fp32 (no bf16 cast): w1_m = W1_s . W_down, w3_m = W3_s . W_down."""
    if layer.kind != "form_E":
        raise SystemExit("form M is defined for MoE layers only")
    down = layer.tensors["block_sparse_moe.routed_expert_down_proj.weight"].float()
    return {"w1_m": layer.tensors["w1_s"].float() @ down, "w3_m": layer.tensors["w3_s"].float() @ down,
            "w2_latent": layer.tensors["w2p"].float(),
            "shared_gate": layer.tensors["block_sparse_moe.shared_experts.gate_proj.weight"].float(),
            "shared_up": layer.tensors["block_sparse_moe.shared_experts.up_proj.weight"].float()}


# --------------------------------------------------------------------------
# charts (c17's convention) and alignment (c17's alignment.json)
# --------------------------------------------------------------------------
@dataclass
class Chart:
    """The two charts of one trunk layer under c17's convention, each under its own name.

    trunk_to_k3 = C_out^T [H x h], bias_to_k3 = C_out_bias [H]:   x_k3 = x @ trunk_to_k3.T + bias_to_k3
    k3_to_trunk = C_in^T  [h x H], bias_to_trunk = C_in_bias [h]: y    = y_k3 @ k3_to_trunk.T + bias_to_trunk
    """

    trunk_layer: int
    path: str
    sha256: str
    arm: str
    trunk_to_k3: torch.Tensor
    bias_to_k3: torch.Tensor
    k3_to_trunk: torch.Tensor
    bias_to_trunk: torch.Tensor
    layer_json_sha256: str | None = None
    fit_trunk_rows_key: str | None = None
    native_boundary_provenance: dict[str, Any] | None = None

    @property
    def k3_hidden(self) -> int:
        return int(self.trunk_to_k3.shape[0])

    @property
    def trunk_hidden(self) -> int:
        return int(self.trunk_to_k3.shape[1])

    def to_k3(self, x: torch.Tensor) -> torch.Tensor:
        """trunk rows -> K3 rows, WITH the input bias (the reference is exact)."""
        return x @ self.trunk_to_k3.T + self.bias_to_k3

    def to_trunk_linear(self, y: torch.Tensor) -> torch.Tensor:
        """K3 rows -> trunk rows WITHOUT the output bias (unrepresentable; recorded, excluded from Y)."""
        return y @ self.k3_to_trunk.T

    def on(self, device: torch.device) -> "Chart":
        return dataclasses.replace(self, trunk_to_k3=self.trunk_to_k3.to(device), bias_to_k3=self.bias_to_k3.to(device),
                                   k3_to_trunk=self.k3_to_trunk.to(device), bias_to_trunk=self.bias_to_trunk.to(device))

    def record(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "arm": self.arm, "layer_json_sha256": self.layer_json_sha256,
                "fit_trunk_rows_key": self.fit_trunk_rows_key,
                "native_boundary_provenance": self.native_boundary_provenance,
                "trunk_to_k3_shape": list(self.trunk_to_k3.shape), "k3_to_trunk_shape": list(self.k3_to_trunk.shape),
                "input_bias_norm": float(self.bias_to_k3.norm()), "output_bias_norm": float(self.bias_to_trunk.norm()),
                "convention": "c17: trunk_to_k3 = C_out^T (+ C_out_bias), k3_to_trunk = C_in^T (+ C_in_bias)"}


def chart_from_c17_tensors(k: int, path: str, sha: str, arm: str, got: dict[str, torch.Tensor], k3_hidden: int,
                           trunk_hidden: int, **extra: Any) -> Chart:
    if "C_in" not in got or "C_out" not in got:
        raise SystemExit(f"{path} must carry C_in [d_k3 x d_trunk] and C_out [d_trunk x d_k3] (c17); has {sorted(got)}")
    c_in = got["C_in"].float()
    c_out = got["C_out"].float()
    if tuple(c_in.shape) != (k3_hidden, trunk_hidden) or tuple(c_out.shape) != (trunk_hidden, k3_hidden):
        raise SystemExit(f"{path}: C_in {tuple(c_in.shape)} / C_out {tuple(c_out.shape)} do not match c17's "
                         f"C_in [d_k3 x d_trunk] / C_out [d_trunk x d_k3] at K3 hidden {k3_hidden} x trunk hidden "
                         f"{trunk_hidden}")
    b_in = got["C_in_bias"].float() if "C_in_bias" in got else torch.zeros(trunk_hidden)
    b_out = got["C_out_bias"].float() if "C_out_bias" in got else torch.zeros(k3_hidden)
    if tuple(b_in.shape) != (trunk_hidden,) or tuple(b_out.shape) != (k3_hidden,):
        raise SystemExit(f"{path}: C_in_bias {tuple(b_in.shape)} / C_out_bias {tuple(b_out.shape)} do not match "
                         f"[d_trunk] / [d_k3]")
    for name, t in (("C_in", c_in), ("C_out", c_out), ("C_in_bias", b_in), ("C_out_bias", b_out)):
        if not bool(torch.isfinite(t).all()):
            raise SystemExit(f"{path}: non-finite chart tensor {name}")
    return Chart(k, path, sha, arm, c_out.T.contiguous(), b_out.contiguous(), c_in.T.contiguous(), b_in.contiguous(),
                 **extra)


def fixture_root_checked(root: str | None, paths: list[str | Path], *, hidden: int, source_hidden: int,
                         device: str = "cpu", layers: int = 1) -> str | None:
    """Explicit tiny mechanics route; never available for campaign geometry/GPU."""
    if root is None:
        return None
    base = Path(root).resolve(strict=True)
    marker = base / FIXTURE_MARKER
    if (device != "cpu" or max(hidden, source_hidden) > 128 or min(hidden, source_hidden) < 1
            or not 0 < layers <= 8 or not marker.is_file()
            or json.loads(marker.read_text()).get("schema") != "neural-forge.c10-synthetic-fixture.v1"):
        raise SystemExit("synthetic fixture route requires marked CPU-only tiny geometry (hidden <=128, layers <=8)")
    if any(not Path(p).resolve().is_relative_to(base) for p in paths if p is not None):
        raise SystemExit("every synthetic fixture input/output must stay beneath its explicit fixture root")
    return str(base)


def native_chart_boundaries(doc: dict[str, Any], layer_json: str, k: int, arm: str, *,
                            trunk_hidden: int, k3_hidden: int) -> dict[str, Any]:
    """Verify observed map endpoints; this validates provenance, never fits maps.

    Each map endpoint names a native capture receipt (path/sha256), a captured
    layer and its tensor key. The receipt must list the SHA-sealed safetensors
    file and its boundaries. Input and write maps use DIFFERENT observed keys.
    Native K3 producer/paired fitter are still required to emit this contract.
    """
    from safetensors import safe_open
    p = (doc.get("arms", {}).get(arm) or {}).get("native_boundary_provenance")
    if not isinstance(p, dict) or p.get("schema") != NATIVE_CHART_SCHEMA or p.get("fixture") is not False:
        raise SystemExit(f"{layer_json}: missing explicit non-fixture native boundary provenance; legacy charts cannot install")
    maps = p.get("maps")
    if not isinstance(maps, dict) or set(maps) != {"C_in", "C_out"}:
        raise SystemExit("native boundary provenance must bind both applied maps")
    expected = {"C_out": (("host", "ffn_input"), ("k3", "ffn_input")),
                "C_in": (("k3", "ffn_output"), ("host", "ffn_output"))}
    receipt_cache, file_cache = {}, {}
    host_identities, evidence_identities, source_layers = set(), set(), set()
    installed_ffns: dict[str, str] = {}
    base = Path(layer_json).parent
    def endpoint(e: Any, family: str, boundary: str) -> dict[str, Any]:
        if not isinstance(e, dict) or e.get("family") != family or e.get("boundary") != boundary:
            raise SystemExit(f"native map endpoint must be {family}.{boundary}; input charts cannot stand in for writes")
        r = e.get("capture_receipt") or {}
        raw_path, sha = r.get("path"), r.get("sha256")
        if not isinstance(raw_path, str) or not raw_path or not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
            raise SystemExit("native endpoint requires capture receipt path and SHA256")
        path = (base / raw_path).resolve(strict=True)
        cache_key = (str(path), sha)
        if cache_key not in receipt_cache:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise SystemExit("native capture receipt SHA256 mismatch")
            receipt_cache[cache_key] = json.loads(raw)
        captured = receipt_cache[cache_key]
        schema = "neural-forge.c17-host-boundaries.v1" if family == "host" else "neural-forge.c17-k3-boundaries.v1"
        code_hashes = captured.get("code_sha256")
        if (captured.get("schema") != schema or captured.get("fixture", False) is not False
                or captured.get("gradients_or_training") is not False
                or not isinstance(code_hashes, dict) or not code_hashes
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(s)) for s in code_hashes.values())):
            raise SystemExit("endpoint must originate in an observed forward-only native capture, not a fixture")
        if family == "k3" and captured.get("stream") != "folded":
            raise SystemExit("K3 endpoints require the accepted folded stream, not teacher or synthetic rows")
        if family == "k3" and captured.get("accepted_prefix_replay_exact") is not True:
            raise SystemExit("K3 capture requires exact accepted-prefix replay evidence")
        if family == "host":
            if captured.get("consumed_tensor_hashes_verified") is not True:
                raise SystemExit("host capture did not verify consumed tensor hashes")
            host_identities.add(captured.get("index_sha256"))
        evidence_identities.add(captured.get("evidence_sha256"))
        layer = e.get("layer")
        if type(layer) is not int or layer < 0 or (family == "host" and layer != k):
            raise SystemExit("native endpoint layer mismatch")
        if family == "k3":
            source_layers.add(layer)
            ffn_sha = (captured.get("installed_ffn_sha256") or {}).get(str(layer))
            if not re.fullmatch(r"[0-9a-f]{64}", str(ffn_sha)) or installed_ffns.get(str(layer), ffn_sha) != ffn_sha:
                raise SystemExit("K3 capture lacks consistent installed FFN identity")
            installed_ffns[str(layer)] = ffn_sha
        entries = [x for x in captured.get("layers", []) if x.get("layer") == layer]
        if len(entries) != 1:
            raise SystemExit("capture receipt does not bind the endpoint layer exactly once")
        entry = entries[0]
        if type(entry.get("rows")) is not int or entry["rows"] < 2:
            raise SystemExit("native captured row count must be an explicit integer >=2")
        row_path = (path.parent / entry["path"]).resolve(strict=True)
        if not row_path.is_relative_to(path.parent):
            raise SystemExit("native captured rows escape the capture directory")
        row_sha = entry.get("sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row_sha)):
            raise SystemExit("native captured row SHA256 missing")
        if str(row_path) not in file_cache:
            file_cache[str(row_path)] = sha256_file(str(row_path))
        if file_cache[str(row_path)] != row_sha:
            raise SystemExit("native captured row SHA256 mismatch")
        width = trunk_hidden if family == "host" else k3_hidden
        with safe_open(str(row_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("schema") != schema or metadata.get("layer") != str(layer):
                raise SystemExit("native captured row schema/layer mismatch")
            if boundary not in handle.keys() or handle.get_slice(boundary).get_shape() != [entry.get("rows"), width]:
                raise SystemExit("native endpoint must name actual captured boundary rows of the expected shape")
            if handle.get_slice(boundary).get_dtype() not in ("F16", "BF16", "F32", "F64"):
                raise SystemExit("native endpoint boundary tensors must be floating-point")
            declared = entry.get("boundaries", {}).get(boundary) or {}
            if declared.get("shape") != [entry.get("rows"), width]:
                raise SystemExit("capture receipt does not declare the actual endpoint boundary")
            identity_keys = ("evidence_row_index", "host_position", "k3_position", "held")
            if not all(name in handle.keys() for name in identity_keys):
                raise SystemExit("native endpoint lacks explicit paired positions and original split")
            identity = {name: handle.get_tensor(name) for name in identity_keys}
            if any(t.dtype != torch.int64 or tuple(t.shape) != (entry.get("rows"),) for t in identity.values()):
                raise SystemExit("native endpoint positions/splits must be int64 vectors")
            if set(identity["held"].tolist()) != {0, 1} or any((t < 0).any() for t in identity.values()):
                raise SystemExit("native endpoints need original fit/held groups and nonnegative positions")
        return {"family": family, "boundary": boundary, "layer": layer, "receipt_sha256": sha,
                "rows_sha256": row_sha, "identity": identity}
    observed = {}
    for name, endpoints in expected.items():
        map_doc = maps[name]
        if not isinstance(map_doc, dict) or set(map_doc) != {"domain", "codomain"}:
            raise SystemExit("each applied map must bind its exact native domain and codomain")
        domains, codomains = map_doc["domain"], map_doc["codomain"]
        domains = domains if isinstance(domains, list) else [domains]
        codomains = codomains if isinstance(codomains, list) else [codomains]
        if not domains or len(domains) != len(codomains):
            raise SystemExit("pooled native map endpoints need one matched pair per source layer")
        pairs = []
        for domain, codomain in zip(domains, codomains):
            left = endpoint(domain, *endpoints[0])
            right = endpoint(codomain, *endpoints[1])
            if any(not torch.equal(left["identity"][key], right["identity"][key]) for key in left["identity"]):
                raise SystemExit("native chart endpoints do not carry identical paired positions/splits")
            for entry in (left, right):
                del entry["identity"]
            pairs.append({"domain": left, "codomain": right})
        observed[name] = pairs
    inputs = [pair["codomain"]["layer"] for pair in observed["C_out"]]
    outputs = [pair["domain"]["layer"] for pair in observed["C_in"]]
    if inputs != outputs or len(inputs) != len(set(inputs)):
        raise SystemExit("native input/write maps must cover identical source layers exactly once")
    if (len(host_identities) != 1 or len(evidence_identities) != 1
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(x)) for x in host_identities | evidence_identities)):
        raise SystemExit("native maps disagree on host checkpoint or evidence identity")
    return {"schema": NATIVE_CHART_SCHEMA, "fixture": False, "maps": observed,
            "host_index_sha256": next(iter(host_identities)), "evidence_sha256": next(iter(evidence_identities)),
            "source_layers": sorted(source_layers), "installed_ffn_sha256": installed_ffns, "provenance_sha256": hashlib.sha256(
                json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def load_chart(charts_dir: str | None, k: int, k3_hidden: int, trunk_hidden: int, identity: bool,
               arm: str = "ridge", *, fixture_root: str | None = None) -> Chart:
    fixture = fixture_root_checked(fixture_root, [charts_dir] if charts_dir else [],
                                   hidden=trunk_hidden, source_hidden=k3_hidden, layers=k + 1)
    if charts_dir is None:
        if fixture is None:
            raise SystemExit("identity charts have no native boundary evidence; explicit tiny fixture path only")
        if not identity:
            raise SystemExit("--charts is required unless --identity-charts (K3 hidden == trunk hidden)")
        if k3_hidden != trunk_hidden:
            raise SystemExit(f"--identity-charts needs K3 hidden {k3_hidden} == trunk hidden {trunk_hidden}")
        eye = torch.eye(trunk_hidden, dtype=torch.float32)
        zero = torch.zeros(trunk_hidden)
        return Chart(k, "identity", sha256_tensor(eye), "identity", eye.clone(), zero.clone(), eye.clone(), zero.clone(),
                     native_boundary_provenance={"fixture": True, "fixture_root": fixture})
    # Capture provenance and chart coefficients are candidate evidence only.
    # No reviewed producer currently measures the exact installed computation
    # against the post-S3 host, accepted source and untouched held evidence.
    # A renamed candidate, affirmative metadata or a self-asserted receipt must
    # therefore have no production allow path. Keep native_chart_boundaries
    # available to the candidate fitter as a provenance-only validator.
    if fixture is None:
        raise SystemExit("production native chart admission is unavailable: a reviewed exact-installed "
                         "evaluator and acceptance policy are required; candidate provenance is not acceptance")
    from safetensors.torch import load_file

    if arm not in CHART_ARM_FILES:
        raise SystemExit(f"--chart-arm must be one of {sorted(CHART_ARM_FILES)}, got {arm!r}")
    layer_dir = os.path.join(charts_dir, f"L{k:02d}")
    path = os.path.join(layer_dir, CHART_ARM_FILES[arm])
    if not os.path.isfile(path):
        raise SystemExit(f"no {arm} chart for trunk layer {k}: c17 writes {path}")
    extra: dict[str, Any] = {}
    layer_json = os.path.join(layer_dir, "layer.json")
    if os.path.isfile(layer_json):
        doc = json.loads(Path(layer_json).read_text(encoding="utf-8"))
        extra["layer_json_sha256"] = sha256_file(layer_json)
        arm_doc = (doc.get("arms") or {}).get(arm) or {}
        convention = arm_doc.get("convention")
        if convention != C17_CONVENTION:
            raise SystemExit(f"{layer_json}: arm {arm!r} records convention {convention!r}; this tool implements "
                             f"{C17_CONVENTION!r}")
        recorded_sha = ((doc.get("files") or {}).get(arm) or {}).get("file_sha256")
        actual_sha = sha256_file(path)
        if recorded_sha != actual_sha:
            raise SystemExit(f"{path}: sha256 {actual_sha} != {recorded_sha} recorded in {layer_json}")
        if type(doc.get("trunk_layer")) is not int or doc["trunk_layer"] != k:
            raise SystemExit(f"{layer_json} is for trunk layer {doc['trunk_layer']}, not {k}")
        extra["fit_trunk_rows_key"] = (doc.get("trunk_taps") or {}).get("rows_key")
        if extra["fit_trunk_rows_key"] not in ("normed", "ffn_input"):
            raise SystemExit(f"{layer_json}: chart was not fitted on native post-norm2 FFN input; prenorm/missing is incompatible")
        extra["native_boundary_provenance"] = ({"fixture": True, "fixture_root": fixture} if fixture else
            native_chart_boundaries(doc, layer_json, k, arm, trunk_hidden=trunk_hidden, k3_hidden=k3_hidden))
    else:
        raise SystemExit(f"{layer_json}: required native chart boundary provenance is missing")
    chart = chart_from_c17_tensors(k, path, actual_sha, arm, load_file(path), k3_hidden, trunk_hidden, **extra)
    return chart


def default_alignment(k3_layers: list[int], n_trunk: int) -> dict[int, list[int]]:
    """D-e1 depth-fraction pairing (== c17.alignment_table): MoE layer l (1..M) -> trunk k with
    l in [1 + floor(M k / n), 1 + floor(M (k+1) / n)); K3 layer 0 joins trunk 0's pool."""
    if n_trunk <= 0:
        raise SystemExit("alignment needs at least one trunk layer")
    moe = [l for l in k3_layers if l != 0]
    m = len(moe)
    table: dict[int, list[int]] = {k: [] for k in range(n_trunk)}
    if 0 in k3_layers:
        table[0].append(0)
    for k in range(n_trunk):
        lo = 1 + (m * k) // n_trunk
        hi = 1 + (m * (k + 1)) // n_trunk
        table[k].extend([l for l in moe if lo <= l < hi])
    return table


def parse_alignment_document(raw: dict[str, Any], path: str) -> tuple[dict[int, list[int]], dict[str, Any]]:
    """c17's ``{schema, summary, table: [{trunk, k3_layers, ...}]}`` or the override form ``{trunk_to_k3: {k: [...]}}``."""
    record: dict[str, Any] = {"source": path, "sha256": sha256_file(path)}
    if "table" in raw:
        schema = raw.get("schema")
        if schema is not None and schema != C17_ALIGNMENT_SCHEMA:
            raise SystemExit(f"{path}: alignment schema {schema!r}, expected {C17_ALIGNMENT_SCHEMA!r}")
        table: dict[int, list[int]] = {}
        for row in raw["table"]:
            k = int(row["trunk"])
            if k in table:
                raise SystemExit(f"{path}: trunk layer {k} appears twice in the alignment table")
            table[k] = [int(x) for x in row["k3_layers"]]
        summary = raw.get("summary") or {}
        if "covered_exactly_once" in summary and not bool(summary["covered_exactly_once"]):
            raise SystemExit(f"{path}: c17 recorded covered_exactly_once = false; refusing the alignment")
        record.update({"schema": schema, "form": "c17 table", "summary": summary})
    elif "trunk_to_k3" in raw:
        table = {int(k): [int(x) for x in v] for k, v in raw["trunk_to_k3"].items()}
        record.update({"form": "trunk_to_k3 override"})
    else:
        raise SystemExit(f"{path}: alignment must be c17's {{schema, summary, table}} or {{trunk_to_k3: {{k: [...]}}}}; "
                         f"has {sorted(raw)}")
    return table, record


def load_alignment(charts_dir: str | None, override: str | None, k3_layers: list[int],
                   n_trunk: int) -> tuple[dict[int, list[int]], dict[str, Any]]:
    path = override or (os.path.join(charts_dir, "alignment.json") if charts_dir else None)
    if path and os.path.isfile(path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        table, record = parse_alignment_document(raw, path)
        for k in range(n_trunk):
            table.setdefault(k, [])
    else:
        if override:
            raise SystemExit(f"--alignment {override} does not exist")
        table = default_alignment(k3_layers, n_trunk)
        record = {"source": "default depth-fraction pairing (D-e1; == c17.alignment_table)"}
    covered = sorted(l for ls in table.values() for l in ls)
    if covered != sorted(k3_layers):
        raise SystemExit(f"alignment must cover every K3 layer exactly once: covered {covered} vs present "
                         f"{sorted(k3_layers)}")
    if sorted(table) != list(range(n_trunk)):
        raise SystemExit(f"alignment keys {sorted(table)} are not the trunk layers 0..{n_trunk - 1}")
    record["trunk_to_k3"] = {str(k): table[k] for k in range(n_trunk)}
    record["trunk_layers_with_two_or_more"] = int(sum(len(v) >= 2 for v in table.values()))
    record["trunk_layers_with_none"] = int(sum(len(v) == 0 for v in table.values()))
    return table, record


# --------------------------------------------------------------------------
# lifted candidate sources
# --------------------------------------------------------------------------
@dataclass
class Source:
    name: str
    kind: str                    # routed | shared | dense | extra
    gate: torch.Tensor           # [M x h] fp32, trunk width
    up: torch.Tensor             # [M x h] fp32, UNscaled
    prior: torch.Tensor          # [h x M] fp32
    activation: str              # situ | silu
    beta: float
    linear_beta: float | None
    origin: dict[str, Any]
    gate_shift: torch.Tensor = field(default_factory=lambda: torch.zeros(0))   # [M] dropped constant w_c . b_in
    up_shift: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    @property
    def width(self) -> int:
        return int(self.gate.shape[0])


def lift_routed_form_e(layer: K3Layer, trunk_to_k3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """gate/up rows of the kept routed channels in trunk width: W1_s . W_down . T (fp32)."""
    down = layer.tensors["block_sparse_moe.routed_expert_down_proj.weight"].float()
    lat = down @ trunk_to_k3                              # [d x h]
    return layer.tensors["w1_s"].float() @ lat, layer.tensors["w3_s"].float() @ lat


def lift_routed_form_m(form_m: dict[str, torch.Tensor], trunk_to_k3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The same rows from form M: w1_m . T (test 6 compares the two)."""
    return form_m["w1_m"].float() @ trunk_to_k3, form_m["w3_m"].float() @ trunk_to_k3


def lift_layer(layer: K3Layer, chart: Chart, rms_mean: float | None) -> list[Source]:
    t_map, u_map, b_in = chart.trunk_to_k3, chart.k3_to_trunk, chart.bias_to_k3
    cfg = layer.cfg
    origin = {"k3_layer": layer.index, "file": layer.path, "sha256": layer.sha256, "chart": chart.path,
              "chart_arm": chart.arm}
    if layer.kind == "dense":
        g_k3 = layer.tensors["mlp.gate_proj.weight"].float()
        u_k3 = layer.tensors["mlp.up_proj.weight"].float()
        prior = u_map @ layer.tensors["mlp.down_proj.weight"].float()
        return [Source(f"K3L{layer.index:02d}.dense", "dense", g_k3 @ t_map, u_k3 @ t_map, prior, "situ", cfg.beta,
                       cfg.linear_beta, {**origin, "lift": "G . T / U . T / U_map . D"}, g_k3 @ b_in, u_k3 @ b_in)]
    if rms_mean is None:
        raise SystemExit("routed prior needs the fit-row mean rms of y_lat")
    gate_r, up_r = lift_routed_form_e(layer, t_map)
    down = layer.tensors["block_sparse_moe.routed_expert_down_proj.weight"].float()
    lat_bias = down @ b_in                                # [d]
    gain = layer.tensors["block_sparse_moe.routed_expert_norm.weight"].float()
    w_up = layer.tensors["block_sparse_moe.routed_expert_up_proj.weight"].float()
    w2p = layer.tensors["w2p"].float()
    # y_lat / rms * g -> W_up -> U, linearised at the MEAN rms.  rms_mean is floored
    # by eps in the caller; an identically-zero routed stream reads 0 here (inv = 0),
    # never 1/0.
    inv = (1.0 / rms_mean) if rms_mean > 0.0 else 0.0
    prior_r = (u_map @ w_up @ (gain.unsqueeze(1) * w2p)) * inv
    sg = layer.tensors["block_sparse_moe.shared_experts.gate_proj.weight"].float()
    su = layer.tensors["block_sparse_moe.shared_experts.up_proj.weight"].float()
    prior_s = u_map @ layer.tensors["block_sparse_moe.shared_experts.down_proj.weight"].float()
    return [
        Source(f"K3L{layer.index:02d}.routed", "routed", gate_r, up_r, prior_r, "situ", cfg.beta, cfg.linear_beta,
               {**origin, "lift": "W1_s . W_down . T / W3_s . W_down . T / U . W_up . diag(g) . W2p / rms_mean",
                "rms_mean": rms_mean, "prior_inverse_rms": inv},
               layer.tensors["w1_s"].float() @ lat_bias, layer.tensors["w3_s"].float() @ lat_bias),
        Source(f"K3L{layer.index:02d}.shared", "shared", sg @ t_map, su @ t_map, prior_s, "situ", cfg.beta,
               cfg.linear_beta, {**origin, "lift": "Wsg . T / Wsu . T / U . Wsd"}, sg @ b_in, su @ b_in),
    ]


def load_extra_block(spec: str, chart_spec: dict[str, str], trunk_hidden: int) -> tuple[int, Source, dict[str, Any]]:
    """``name=path@trunk_layer``: a coreset block MLP (gate/up/down [+ metadata activation])
    pushed into the paired trunk FFN rows through its own c17-convention chart (the S7 blend arm)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if "=" not in spec or "@" not in spec:
        raise SystemExit(f"--extra-blocks expects name=path@trunk_layer, got {spec!r}")
    name, rest = spec.split("=", 1)
    path, layer_text = rest.rsplit("@", 1)
    k = int(layer_text)
    got = load_file(path)
    for key in ("gate", "up", "down"):
        if key not in got:
            raise SystemExit(f"{path}: extra block needs gate/up/down; has {sorted(got)}")
    with safe_open(path, "pt") as handle:
        meta = handle.metadata() or {}
    activation = str(meta.get("activation", "silu"))
    beta = float(meta.get("situ_gate_beta", 4.0))
    lb_raw = meta.get("situ_up_beta")
    linear_beta = None if lb_raw in (None, "", "inf") else float(lb_raw)
    if name not in chart_spec:
        raise SystemExit(f"--extra-charts carries no chart for extra block {name!r}")
    chart_file = chart_spec[name]
    block_hidden = int(got["gate"].shape[1])
    chart = chart_from_c17_tensors(k, chart_file, sha256_file(chart_file), "extra", load_file(chart_file), block_hidden,
                                   trunk_hidden)
    g, u = got["gate"].float(), got["up"].float()
    source = Source(f"extra.{name}", "extra", g @ chart.trunk_to_k3, u @ chart.trunk_to_k3,
                    chart.k3_to_trunk @ got["down"].float(), activation, beta, linear_beta,
                    {"file": path, "sha256": sha256_file(path), "chart": chart_file, "chart_sha256": chart.sha256,
                     "trunk_layer": k, "output_bias_norm": float(chart.bias_to_trunk.norm())},
                    g @ chart.bias_to_k3, u @ chart.bias_to_k3)
    record = {"name": name, "path": path, "sha256": source.origin["sha256"], "trunk_layer": k,
              "width": source.width, "activation": activation}
    return k, source, record


# --------------------------------------------------------------------------
# SiTU -> SiLU per-channel scalars
# --------------------------------------------------------------------------
def situ_to_silu_scalars(gate: torch.Tensor, up: torch.Tensor, x_rows: torch.Tensor, beta: float,
                         linear_beta: float | None, device: torch.device, chunk: int = 1024) -> dict[str, Any]:
    """Least-squares scalar s_c with silu(g.x)(s u.x) ~ situ(g.x, u.x) over ``x_rows``.

    Closed form per channel: s = <a, b> / <a, a>, ||s a - b||^2 = s^2 <a,a> - 2 s <a,b> + <b,b>.
    A channel with <a, a> = 0 is dead on these rows; s = 1 is installed (it changes
    nothing) and the channel is counted.
    """
    g = gate.to(device)
    u = up.to(device)
    m = int(g.shape[0])
    aa = torch.zeros(m, dtype=torch.float64, device=device)
    ab = torch.zeros(m, dtype=torch.float64, device=device)
    bb = torch.zeros(m, dtype=torch.float64, device=device)
    for start in range(0, int(x_rows.shape[0]), chunk):
        x = x_rows[start:start + chunk].to(device=device, dtype=torch.float32)
        pre_g = x @ g.T
        pre_u = x @ u.T
        a = F.silu(pre_g) * pre_u
        b = glu("situ", pre_g, pre_u, beta, linear_beta)
        aa += (a.double() * a.double()).sum(0)
        ab += (a.double() * b.double()).sum(0)
        bb += (b.double() * b.double()).sum(0)
    dead = aa <= 0.0
    scale = torch.where(dead, torch.ones_like(aa), ab / torch.where(dead, torch.ones_like(aa), aa))
    sse = scale * scale * aa - 2.0 * scale * ab + bb
    sse = sse.clamp_min(0.0)
    per_channel = torch.where(bb > 0, torch.sqrt(sse / torch.where(bb > 0, bb, torch.ones_like(bb))),
                              torch.zeros_like(bb))
    total_b = float(bb.sum())
    rel = math.sqrt(float(sse.sum()) / total_b) if total_b > 0 else 0.0
    return {"scale": scale.float().cpu(), "relL2": rel, "per_channel_relL2_max": float(per_channel.max()) if m else 0.0,
            "per_channel_relL2_mean": float(per_channel.mean()) if m else 0.0,
            "dead_channels": int(dead.sum()), "channels": m, "rows": int(x_rows.shape[0]),
            "scale_min": float(scale.min()) if m else None, "scale_max": float(scale.max()) if m else None,
            "definition": "s_c = <silu(g.x)(u.x), situ(g.x,u.x)> / ||silu(g.x)(u.x)||^2 over the fit rows; "
                          "relL2 = ||s a - b||_F / ||b||_F over every channel of the source"}


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------
@dataclass
class Evidence:
    path: str
    sha256: str
    manifest_sha256: str
    batches: list[tuple[int, torch.Tensor]]
    split: dict[int, str]
    check_rows: torch.Tensor      # over the concatenated fit rows
    sequence_length: int

    @property
    def fit_batches(self) -> list[int]:
        return [b for b, _ in self.batches if self.split[b] == "fit"]

    @property
    def held_batches(self) -> list[int]:
        return [b for b, _ in self.batches if self.split[b] == "held"]


def load_evidence(path: str) -> Evidence:
    manifest = qf_corpus.read_manifest(path)
    batches = qf_corpus.corpus(manifest)
    split = qf_corpus.split_by_batch(manifest)
    check = qf_corpus.fit_check_rows(manifest)
    return Evidence(path, sha256_file(path), str(manifest["manifest_sha256"]), batches, split, check,
                    int(manifest["sequence_length"]))


# --------------------------------------------------------------------------
# the trunk engines: resident (test only) and streamed (the composite's cards)
# --------------------------------------------------------------------------
class Capture:
    """Forward pre-hooks on ``ffn`` modules: the FFN input (post norm2) per token."""

    def __init__(self) -> None:
        self.rows: dict[int, dict[str, list[torch.Tensor]]] = {}
        self.handles: list[Any] = []
        self.current = "fit"

    def arm(self, k: int, ffn: torch.nn.Module) -> None:
        self.rows.setdefault(k, {"fit": [], "held": []})
        self.handles.append(ffn.register_forward_pre_hook(self._hook(k)))

    def _hook(self, k: int):
        def fn(_module, inputs):
            x = inputs[0]
            self.rows[k][self.current].append(x.detach().reshape(-1, x.shape[-1]).float().cpu())
        return fn

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def concatenated(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        fit = torch.cat(self.rows[k]["fit"]) if self.rows[k]["fit"] else torch.zeros(0, 0)
        held = torch.cat(self.rows[k]["held"]) if self.rows[k]["held"] else torch.zeros(0, fit.shape[1] if fit.ndim == 2 else 0)
        return fit, held

    def collect(self, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        out = {}
        for k in layers:
            fit, held = self.concatenated(k)
            out[k] = {"fit": fit, "held": held}
        return out


def _model_dtype(views: dict[str, Any], requested: torch.dtype | None) -> torch.dtype:
    """``load_chimera_checkpoint``'s rule: the checkpoint's embed dtype unless asked."""
    if requested is not None:
        return requested
    from chimera_omni.importing.safetensors_io import _torch_dtype
    anchor = views.get("embed.weight")
    dtype = _torch_dtype(anchor.dtype) if anchor is not None else torch.float32
    return dtype if dtype.is_floating_point else torch.float32


class ResidentEngine:
    """The whole model on ``device`` (test only; proven byte-identical to the streamed engine)."""

    name = "resident"

    def __init__(self, trunk_dir: Path, trunk_cfg: ChimeraConfig, views: dict[str, Any], device: torch.device,
                 dtype: torch.dtype | None, evidence: Evidence) -> None:
        self.device = device
        self.evidence = evidence
        self.cfg = trunk_cfg
        self.model, _cfg, self.load_record = load_chimera_checkpoint(trunk_dir, device=str(device), dtype=dtype)
        self.model.eval()
        self.ffn_dtype = self.model.blocks[0].ffn.up.weight.dtype
        self.boundary = 0

    def record(self) -> dict[str, Any]:
        return {"engine": self.name, "state_sha256": self.load_record["state_sha256"], "dtype": self.load_record["dtype"],
                "dtypes_on_disk": self.load_record["dtypes_on_disk"]}

    def capture(self, group: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        capture = Capture()
        for k in group:
            capture.arm(k, self.model.blocks[k].ffn)
        try:
            with torch.no_grad():
                for batch_index, ids in self.evidence.batches:
                    capture.current = self.evidence.split[batch_index]
                    self.model(ids.to(self.device))
        finally:
            capture.close()
        return capture.collect(group)

    def install(self, k: int, up_weight: torch.Tensor, down_weight: torch.Tensor) -> None:
        width = int(down_weight.shape[1])
        ffn = FeedForward(self.cfg, intermediate_size=width).to(device=self.device, dtype=self.ffn_dtype)
        with torch.no_grad():
            ffn.up.weight.copy_(up_weight.to(device=self.device, dtype=self.ffn_dtype))
            ffn.down.weight.copy_(down_weight.to(device=self.device, dtype=self.ffn_dtype))
        self.model.blocks[k].ffn = ffn

    def advance(self, group: list[int]) -> None:
        self.boundary = group[-1] + 1

    def close(self) -> None:
        del self.model
        gc.collect()


class StreamedEngine:
    """One block on the device at a time; the residual stream on the host in the model dtype.

    Reproduces ``ChimeraOmniModel._forward_trunk`` for ``residual_mode == "none"`` and
    text-only input (no media, all tokens valid) block by block: embed -> [where(valid)]
    -> block(x, token_valid_mask=valid, no media masks) -> where(valid) ... exactly the ops
    the resident model runs, in the same dtype, so the FFN-input rows and the residual
    are bit-identical to the resident path.
    """

    name = "streamed"

    def __init__(self, trunk_dir: Path, trunk_cfg: ChimeraConfig, views: dict[str, Any], device: torch.device,
                 dtype: torch.dtype | None, evidence: Evidence) -> None:
        if trunk_cfg.residual_mode != "none":
            raise SystemExit(f"--engine streamed reproduces residual_mode 'none' (the S0/S3 trunks); this trunk is "
                             f"{trunk_cfg.residual_mode!r} -- use --engine resident on a card that holds it")
        if any(trunk_cfg.block_kind(i) == "charted" for i in range(trunk_cfg.num_layers)):
            raise SystemExit("--engine streamed: the trunk carries charted blocks; C10 runs before S7")
        self.device = device
        self.evidence = evidence
        self.cfg = trunk_cfg
        self.views = views
        self.dtype = _model_dtype(views, dtype)
        self.ffn_dtype = self.dtype
        self.installed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.boundary = 0
        self.residual: dict[int, torch.Tensor] = {}
        self.valid: dict[int, torch.Tensor] = {}
        self.dtypes_on_disk = sorted({v.dtype for v in views.values()})
        embed_weight = views["embed.weight"].torch(device=str(device), dtype=self.dtype)
        with torch.no_grad():
            for batch_index, ids in evidence.batches:
                ids_d = ids.to(device)
                valid = torch.ones(ids_d.shape, dtype=torch.bool, device=device)
                x = F.embedding(ids_d, embed_weight)
                x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
                self.residual[batch_index] = x.cpu()
                self.valid[batch_index] = valid.cpu()
        del embed_weight

    def record(self) -> dict[str, Any]:
        return {"engine": self.name, "dtype": str(self.dtype).replace("torch.", ""),
                "dtypes_on_disk": self.dtypes_on_disk,
                "residual_cache_bytes": int(sum(t.numel() * t.element_size() for t in self.residual.values())),
                "rule": "host residual in the model dtype; one block resident on --device; capture with the group's "
                        "TRUNK FFNs from the group boundary, then advance through the group's NEW FFNs"}

    def block(self, k: int, use_installed: bool) -> ChimeraBlock:
        prefix = f"blocks.{k}."
        block = ChimeraBlock(self.cfg, k).to(device=self.device, dtype=self.dtype)
        if use_installed and k in self.installed:
            up_weight, down_weight = self.installed[k]
            block.ffn = FeedForward(self.cfg, intermediate_size=int(down_weight.shape[1])).to(
                device=self.device, dtype=self.ffn_dtype)
        target = block.state_dict()
        state: dict[str, torch.Tensor] = {}
        for name, reference in target.items():
            full = prefix + name
            if use_installed and k in self.installed and name in ("ffn.up.weight", "ffn.down.weight"):
                continue
            view = self.views.get(full)
            if view is None:
                raise KeyError(f"trunk checkpoint has no tensor {full!r} for block {k}")
            state[name] = view.torch(device=str(self.device),
                                     dtype=self.dtype if reference.is_floating_point() else reference.dtype)
        if use_installed and k in self.installed:
            up_weight, down_weight = self.installed[k]
            state["ffn.up.weight"] = up_weight.to(device=self.device, dtype=self.ffn_dtype)
            state["ffn.down.weight"] = down_weight.to(device=self.device, dtype=self.ffn_dtype)
        block.load_state_dict(state, strict=True)
        block.eval()
        return block

    def _run(self, block: ChimeraBlock, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # keyword form: the c5e6052 tree's block takes 5 arguments, main's takes 7 (charted_*)
        x = block(x, native_media_slot_mask=None, valid_media_mask=None, enable_native_media_refinement=False,
                  token_valid_mask=valid)
        return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))

    def capture(self, group: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        if group[0] != self.boundary:
            raise SystemExit(f"streamed capture of group {group} needs the residual at boundary {group[0]}, "
                             f"it is at {self.boundary}")
        temp = dict(self.residual)
        capture = Capture()
        try:
            with torch.no_grad():
                for k in group:
                    block = self.block(k, use_installed=False)     # the group's TRUNK FFNs (ordering law)
                    capture.arm(k, block.ffn)
                    for batch_index, _ids in self.evidence.batches:
                        capture.current = self.evidence.split[batch_index]
                        x = temp[batch_index].to(self.device)
                        temp[batch_index] = self._run(block, x, self.valid[batch_index].to(self.device)).cpu()
                    capture.close()
                    del block
                    gc.collect()
        finally:
            capture.close()
        del temp
        return capture.collect(group)

    def install(self, k: int, up_weight: torch.Tensor, down_weight: torch.Tensor) -> None:
        self.installed[k] = (up_weight.to(self.ffn_dtype).cpu().contiguous(), down_weight.to(self.ffn_dtype).cpu().contiguous())

    def advance(self, group: list[int]) -> None:
        if group[0] != self.boundary:
            raise SystemExit(f"streamed advance through {group} from boundary {self.boundary}: groups must be contiguous")
        missing = [k for k in group if k not in self.installed]
        if missing:
            raise SystemExit(f"streamed advance through {group}: layers {missing} are not installed")
        with torch.no_grad():
            for k in group:
                block = self.block(k, use_installed=True)
                for batch_index, _ids in self.evidence.batches:
                    x = self.residual[batch_index].to(self.device)
                    self.residual[batch_index] = self._run(block, x, self.valid[batch_index].to(self.device)).cpu()
                del block
                gc.collect()
        self.boundary = group[-1] + 1

    def close(self) -> None:
        self.residual = {}
        gc.collect()


ENGINES = {"streamed": StreamedEngine, "resident": ResidentEngine}


# --------------------------------------------------------------------------
# parent MLP targets
# --------------------------------------------------------------------------
@dataclass
class ParentMlp:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    activation: str
    beta: float
    linear_beta: float | None
    sha256: dict[str, str]

    def __call__(self, x: torch.Tensor, chunk: int) -> torch.Tensor:
        out = torch.empty(x.shape[0], self.down.shape[0], dtype=torch.float32, device=x.device)
        for start in range(0, x.shape[0], chunk):
            piece = x[start:start + chunk]
            out[start:start + chunk] = glu(self.activation, piece @ self.gate.T, piece @ self.up.T, self.beta,
                                           self.linear_beta) @ self.down.T
        return out


def load_parent_mlp(parent_dir: str, k: int, parent_cfg: ChimeraConfig, device: torch.device) -> ParentMlp:
    names = [f"blocks.{k}.ffn.up.weight", f"blocks.{k}.ffn.down.weight"]
    views = read_safetensors(parent_dir, names, verify=True)
    up_full = views[names[0]].torch(device="cpu", dtype=torch.float32)
    down = views[names[1]].torch(device="cpu", dtype=torch.float32)
    inter = int(up_full.shape[0]) // 2
    return ParentMlp(up_full[:inter].to(device), up_full[inter:].to(device), down.to(device), parent_cfg.activation_kind,
                     float(parent_cfg.situ_gate_beta), float(parent_cfg.situ_up_beta),
                     {n: views[n].sha256() for n in names})


# --------------------------------------------------------------------------
# Gram accumulation (host fp64, device column blocks)
# --------------------------------------------------------------------------
def _gram_add(target: torch.Tensor, a: torch.Tensor, block: int) -> None:
    p = int(a.shape[1])
    for i in range(0, p, block):
        ai = a[:, i:i + block]
        for j in range(i, p, block):
            prod = (ai.T @ a[:, j:j + block]).double().cpu()
            target[i:i + block, j:j + block] += prod
            if j != i:
                target[j:j + block, i:i + block] += prod.T


def accumulate_gram(features: Callable[[torch.Tensor], torch.Tensor], x_rows: torch.Tensor, y_rows: torch.Tensor,
                    row_mask: torch.Tensor | None, width: int, device: torch.device, chunk: int,
                    block: int) -> dict[str, Any]:
    """ONE Gram G = A^T A [width x width] fp64 and B = A^T Y over the rows where ``row_mask``
    is true (all rows when None), skipping non-finite feature/target rows; returns the
    finite-row mask so a second pass can restrict to exactly the same rows."""
    h = int(y_rows.shape[1])
    gram = torch.zeros(width, width, dtype=torch.float64)
    cross = torch.zeros(width, h, dtype=torch.float64)
    y_sq = 0.0
    dropped = 0
    n = int(x_rows.shape[0])
    finite_rows = torch.ones(n, dtype=torch.bool)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        select = row_mask[start:stop] if row_mask is not None else None
        if select is not None and not bool(select.any()):
            continue
        x = x_rows[start:stop]
        y = y_rows[start:stop]
        if select is not None:
            x, y = x[select], y[select]
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        a = features(x)
        finite = torch.isfinite(a).all(1) & torch.isfinite(y).all(1)
        if not bool(finite.all()):
            dropped += int((~finite).sum())
            if select is not None:
                idx = torch.nonzero(select).flatten()[~finite.cpu()]
                finite_rows[start + idx] = False
            else:
                finite_rows[start:stop] = finite.cpu()
            a, y = a[finite], y[finite]
        _gram_add(gram, a, block)
        cross += (a.T @ y).double().cpu()
        y_sq += float((y.double() * y.double()).sum())
        del a, x, y
    return {"gram": gram, "cross": cross, "y_sq": y_sq, "dropped": dropped, "finite_rows": finite_rows}


# --------------------------------------------------------------------------
# selection: Qwen rows pinned, K3 pool pivoted on its Schur complement
# --------------------------------------------------------------------------
def select_k3_channels(g_all: torch.Tensor, q: int, k3: int, block: int = 4096) -> tuple[torch.Tensor, dict[str, Any]]:
    p = int(g_all.shape[0]) - q
    if k3 <= 0 or p == 0:
        return torch.zeros(0, dtype=torch.long), {"selected": 0, "pool": p, "rank_starved": False, "filled": 0}
    if not bool(torch.isfinite(g_all).all()):
        lam = RIDGE_REL * float(torch.diagonal(g_all).mean())
        report = qf_solve.gram_defect_report(g_all, lam, RIDGE_REL)
        raise qf_solve.GramNotPositiveDefinite(qf_solve._defect_message(report, "c10 union selection Gram"), report)
    gqq = g_all[:q, :q]
    gqk = g_all[:q, q:]
    # the Schur complement is formed IN PLACE into one pool-sized buffer, row block by
    # row block: never gqk.T @ solve AND gkk - (...) as two extra p x p matrices
    schur = g_all[q:, q:].clone()
    if q > 0:
        lam = RIDGE_REL * float(torch.diagonal(gqq).mean())
        regularised = gqq.clone()
        regularised.diagonal().add_(lam)
        try:
            factor = torch.linalg.cholesky(regularised)
        except Exception as error:
            report = qf_solve.gram_defect_report(gqq, lam, RIDGE_REL)
            report["cholesky_error"] = f"{type(error).__name__}: {error}"
            raise qf_solve.GramNotPositiveDefinite(
                qf_solve._defect_message(report, "c10 union selection: the pinned Qwen block"), report) from error
        del regularised
        solved = torch.cholesky_solve(gqk, factor)         # [q x p]
        del factor
        for i in range(0, p, block):
            schur[i:i + block].sub_(gqk[:, i:i + block].T @ solved)
        del solved
        mode = "pivoted Cholesky on the Schur complement of the K3 pool given the pinned Qwen rows (formed in place)"
    else:
        mode = "pivoted Cholesky on the K3 pool Gram (no Qwen rows)"
    diag_energy = float(torch.diagonal(schur).clamp_min(0).sum())
    chosen = c14.pivoted_columns_gram(schur, min(k3, p))
    del schur
    starved = int(chosen.numel()) < min(k3, p)
    filled = 0
    if starved:
        mask = torch.ones(p, dtype=torch.bool)
        mask[chosen] = False
        rest = torch.nonzero(mask).flatten()[: min(k3, p) - int(chosen.numel())]
        filled = int(rest.numel())
        chosen = torch.sort(torch.cat([chosen, rest])).values
    return chosen, {"selected": int(chosen.numel()), "pool": p, "requested": k3, "mode": mode,
                    "rank_starved": bool(starved), "filled_with_rank_starved_rows": filled,
                    "schur_diagonal_energy": diag_energy,
                    "note": ("the pool's Schur diagonal carries no energy: the filled rows are rows in name "
                             "only and the layer's veto records it" if starved else "")}


# --------------------------------------------------------------------------
# size gate
# --------------------------------------------------------------------------
def header_total(root: str | Path) -> tuple[int, dict[str, int]]:
    counts = oc_seal.safetensors_tensor_counts(root)
    return int(sum(counts.values())), counts


def resolve_parent_parameter_count(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.parent_parameter_count is not None:
        return int(args.parent_parameter_count), {"source": "--parent-parameter-count"}
    if args.parent_record:
        record = json.loads(Path(args.parent_record).read_text(encoding="utf-8"))
        body = record.get("record", record)
        count = body.get("parameter_count")
        if count is None:
            raise SystemExit(f"{args.parent_record} carries no parameter_count")
        return int(count), {"source": args.parent_record, "sha256": sha256_file(args.parent_record),
                            "checkpoint_id": body.get("checkpoint_id")}
    total, _ = header_total(args.parent_trunk)
    return total, {"source": "the --parent-trunk headers (oc_seal.safetensors_tensor_counts)"}


def refuse_size(child_total: int, parent_total: int, where: str, out: Path | None = None) -> None:
    if child_total >= parent_total:
        if out is not None and out.is_dir():
            shutil.rmtree(out)
        raise SystemExit(
            f"REFUSED ({where}): the child's header count {child_total:,} is not below the parent record's "
            f"parameter_count {parent_total:,} (excess {child_total - parent_total:,}); rebuild with a smaller "
            f"--k3-channels" + (f"; {out} removed" if out is not None else ""))


# --------------------------------------------------------------------------
# the per-layer transaction
# --------------------------------------------------------------------------
@dataclass
class LayerResult:
    up_weight: torch.Tensor      # [2W x h] in the FFN dtype -- what is installed, scored and shipped
    down_weight: torch.Tensor    # [h x W]  in the FFN dtype
    record: dict[str, Any]


def _shift_relL2(shift: torch.Tensor, rows: torch.Tensor, xtx: torch.Tensor, n: int) -> float:
    """||1 shift^T||_F / ||X W^T||_F over the fit rows, from X^T X (fp64, [h x h])."""
    if shift.numel() == 0 or n == 0:
        return 0.0
    w = rows.double()
    energy = float((w @ xtx * w).sum())
    num = math.sqrt(float(shift.double().pow(2).sum()) * n)
    if energy <= 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / math.sqrt(energy)


def install_layer(k: int, x_fit: torch.Tensor, x_held: torch.Tensor, trunk_up: torch.Tensor, trunk_down: torch.Tensor,
                  trunk_cfg: ChimeraConfig, parent: ParentMlp, k3_layers: list[K3Layer], chart: Chart | None,
                  extras: list[Source], evidence: Evidence, k3_channels: int, device: torch.device,
                  chunk: int, block: int, dump_dir: str | None, ffn_dtype: torch.dtype = torch.float32,
                  chart_bias_ceiling: float = 0.1) -> LayerResult:
    started = time.time()
    q = int(trunk_up.shape[0]) // 2
    h = int(trunk_up.shape[1])
    gate_q = trunk_up[:q].float()
    up_q = trunk_up[q:].float()
    record: dict[str, Any] = {"layer": k, "qwen_rows": q, "hidden": h,
                              "k3_layers": [l.index for l in k3_layers], "chart": None,
                              "rows": {"fit": int(x_fit.shape[0]), "held": int(x_held.shape[0])}}
    if chart is not None:
        record["chart"] = chart.record()

    # -- finite rows only (section 3b: a NaN row poisons the whole Gram) ----
    def finite_rows(x: torch.Tensor, label: str) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.isfinite(x).all(1)
        dropped = int((~mask).sum())
        record["rows"][f"{label}_nonfinite_dropped"] = dropped
        if x.shape[0] and dropped / x.shape[0] > NONFINITE_CEILING:
            raise SystemExit(f"layer {k}: {dropped} of {x.shape[0]} captured {label} rows are non-finite "
                             f"(> {NONFINITE_CEILING:.0%}); the capture, not the fold, is wrong")
        return x[mask], mask

    check = evidence.check_rows
    if int(check.numel()) != int(x_fit.shape[0]):
        raise SystemExit(f"layer {k}: check mask is {int(check.numel())} rows, captured fit rows are "
                         f"{int(x_fit.shape[0])}")
    x_fit, fit_mask = finite_rows(x_fit, "fit")
    x_held, _held_mask = finite_rows(x_held, "held")
    check = check[fit_mask]          # the check split follows the rows that survived
    if x_fit.shape[0] == 0:
        raise SystemExit(f"layer {k}: no fit rows")

    # -- targets ---------------------------------------------------------------
    with torch.no_grad():
        y_fit = parent(x_fit.to(device), chunk).cpu()
        y_held = parent(x_held.to(device), chunk).cpu() if x_held.shape[0] else torch.zeros(0, h)
        term_fit = torch.zeros_like(y_fit)
        term_held = torch.zeros_like(y_held)
        per_k3: list[dict[str, Any]] = []
        sources: list[Source] = []
        chart_dev = chart.on(device) if chart is not None else None
        for layer in k3_layers:
            assert chart is not None and chart_dev is not None
            dev = layer.device_tensors(device)             # ONE device copy per install (minor finding)
            rms_sum, rms_rows = 0.0, 0
            for label, x_rows, term in (("fit", x_fit, term_fit), ("held", x_held, term_held)):
                for start in range(0, int(x_rows.shape[0]), chunk):
                    xk = chart_dev.to_k3(x_rows[start:start + chunk].to(device))       # exact: with C_out_bias
                    out_k3, y_lat = layer.reference(xk, dev)
                    term[start:start + chunk] += chart_dev.to_trunk_linear(out_k3).cpu()  # C_in_bias excluded, recorded
                    if label == "fit" and y_lat is not None:
                        rms_sum += float(torch.sqrt(y_lat.double().pow(2).mean(1) + layer.cfg.rms_norm_eps).sum())
                        rms_rows += int(y_lat.shape[0])
            del dev
            rms_mean = (rms_sum / rms_rows) if rms_rows else None
            if layer.kind == "form_E" and rms_mean is not None:
                rms_mean = max(rms_mean, math.sqrt(layer.cfg.rms_norm_eps))
            lifted = lift_layer(layer, chart, rms_mean)
            sources.extend(lifted)
            per_k3.append({"k3_layer": layer.index, "kind": layer.kind, "file": layer.path, "sha256": layer.sha256,
                           "routed_width": layer.routed_width, "shared_width": layer.shared_width,
                           "dense_width": layer.dense_width, "y_lat_rms_mean_fit": rms_mean,
                           "zero_gains": layer.zero_gains()})
        sources.extend(extras)
        y_fit = y_fit + term_fit
        y_held = y_held + term_held
    record["k3_sources"] = per_k3
    record["extra_sources"] = [{"name": s.name, "width": s.width, **s.origin} for s in extras]
    fit_energy = float(term_fit.norm()) / max(float(y_fit.norm()), 1e-30)
    held_energy = (float(term_held.norm()) / max(float(y_held.norm()), 1e-30)) if y_held.shape[0] else None
    record["k3_term_energy_fraction"] = {"fit": fit_energy, "held": held_energy,
                                         "definition": "||sum_l FFN_k3,l(T x + b) U^T|| / ||Y||, Y = MLP_parent(x) + that sum"}
    finite_targets = torch.isfinite(y_fit).all(1)
    if not bool(finite_targets.all()):
        dropped = int((~finite_targets).sum())
        record["rows"]["fit_target_nonfinite_dropped"] = dropped
        if dropped / y_fit.shape[0] > NONFINITE_CEILING:
            raise SystemExit(f"layer {k}: {dropped} of {y_fit.shape[0]} fit targets are non-finite (> "
                             f"{NONFINITE_CEILING:.0%}) -- the K3 reference or the parent MLP produced NaN/Inf; "
                             "refused before any Gram (section 3b)")
        x_fit, y_fit, check = x_fit[finite_targets], y_fit[finite_targets], check[finite_targets]
    if y_held.shape[0]:
        finite_held = torch.isfinite(y_held).all(1)
        record["rows"]["held_target_nonfinite_dropped"] = int((~finite_held).sum())
        x_held, y_held = x_held[finite_held], y_held[finite_held]

    # -- the chart biases, out loud ------------------------------------------------
    n_fit = int(x_fit.shape[0])
    xtx = torch.zeros(h, h, dtype=torch.float64)
    for start in range(0, n_fit, chunk):
        piece = x_fit[start:start + chunk].to(device=device, dtype=torch.float32)
        xtx += (piece.T @ piece).double().cpu()
    bias_records = []
    worst = 0.0
    for s in sources:
        g_rel = _shift_relL2(s.gate_shift, s.gate, xtx, n_fit)
        u_rel = _shift_relL2(s.up_shift, s.up, xtx, n_fit)
        worst = max(worst, g_rel, u_rel)
        bias_records.append({"source": s.name, "gate_shift_relL2": g_rel, "up_shift_relL2": u_rel,
                             "gate_shift_norm": float(s.gate_shift.norm()) if s.gate_shift.numel() else 0.0,
                             "up_shift_norm": float(s.up_shift.norm()) if s.up_shift.numel() else 0.0})
    del xtx
    out_bias = torch.zeros(h) if chart is None else chart.bias_to_trunk * len(k3_layers)
    out_rel = (float(out_bias.norm()) * math.sqrt(n_fit) / max(float(y_fit.norm()), 1e-30)) if chart is not None else 0.0
    record["chart_bias"] = {
        "input": {"treatment": "applied exactly in the K3 reference; DROPPED from the lifted rows (a bias-free "
                               "FeedForward row cannot carry a constant pre-activation shift); the ridge refit "
                               "absorbs what it can", "per_source": bias_records, "worst_shift_relL2": worst,
                  "definition": "||1 (w . b_in)^T||_F / ||X W^T||_F over the fit rows, per source and gate/up"},
        "output": {"treatment": "UNREPRESENTABLE by the union FFN (a constant vector per token); excluded from Y, "
                                "reported", "norm": float(out_bias.norm()), "energy_fraction_fit": out_rel,
                   "definition": "||C_in_bias|| sqrt(n_fit) / ||Y_fit||, summed over the layer's K3 layers"},
        "ceiling": chart_bias_ceiling,
    }
    if chart_bias_ceiling > 0 and (worst > chart_bias_ceiling or out_rel > chart_bias_ceiling):
        raise SystemExit(f"REFUSED layer {k}: chart biases are not negligible (worst dropped input shift relL2 "
                         f"{worst:.4g}, output bias energy fraction {out_rel:.4g} > --chart-bias-ceiling "
                         f"{chart_bias_ceiling:g}); refit the charts with --no-fit-bias in c17 or raise the ceiling "
                         f"knowingly")

    # -- SiTU -> SiLU scalars, then the pool ------------------------------------
    situ_records = []
    pool_gate, pool_up, pool_prior, pool_owner = [], [], [], []
    for index, s in enumerate(sources):
        if s.activation == "situ":
            fit_scalar = situ_to_silu_scalars(s.gate, s.up, x_fit, s.beta, s.linear_beta, device, chunk)
            scale = fit_scalar.pop("scale")
        else:
            scale = torch.ones(s.width)
            fit_scalar = {"relL2": 0.0, "channels": s.width, "note": "already SiLU; no scalar"}
        situ_records.append({"source": s.name, "kind": s.kind, **fit_scalar})
        pool_gate.append(s.gate)
        pool_up.append(s.up * scale.unsqueeze(1))
        pool_prior.append(s.prior)
        pool_owner.extend([(index, j) for j in range(s.width)])
    record["situ_to_silu"] = situ_records
    if pool_gate:
        gate_p = torch.cat(pool_gate)
        up_p = torch.cat(pool_up)
        prior_p = torch.cat(pool_prior, dim=1)
    else:
        gate_p = torch.zeros(0, h)
        up_p = torch.zeros(0, h)
        prior_p = torch.zeros(h, 0)
    p = int(gate_p.shape[0])
    k3_share = min(k3_channels, p)
    record["pool"] = {"qwen": q, "k3_candidates": p, "k3_share_requested": k3_channels, "k3_share": k3_share,
                      "per_source": [{"source": s.name, "width": s.width} for s in sources]}
    if k3_share < k3_channels:
        record["pool"]["note"] = (f"the pool ({p}) is narrower than --k3-channels {k3_channels}; the layer is "
                                  f"padded with zero rows to keep intermediate_sizes uniform")

    gate_all = torch.cat([gate_q, gate_p]).to(device)
    up_all = torch.cat([up_q, up_p]).to(device)

    def features(x: torch.Tensor) -> torch.Tensor:
        return F.silu(x @ gate_all.T) * (x @ up_all.T)

    # -- ONE pool-sized Gram over every fit row --------------------------------------
    full = accumulate_gram(features, x_fit, y_fit, None, q + p, device, chunk, block)
    record["rows"]["fit_feature_nonfinite_dropped"] = full["dropped"]
    finite_rows = full["finite_rows"]
    record["gram"] = {"width": q + p, "dtype": "fp64 host accumulator, fp32 column-block products on --device",
                      "check_rows": int((check & finite_rows).sum()), "inner_rows": int((~check & finite_rows).sum()),
                      "rule": "one pool-sized Gram; the check Gram is accumulated on the KEPT columns only, after "
                              "the full Gram is sliced and freed"}

    # -- selection -------------------------------------------------------------
    chosen, selection = select_k3_channels(full["gram"], q, k3_share)
    kept = torch.cat([torch.arange(q, dtype=torch.long), q + chosen])
    per_source_kept: dict[str, int] = {}
    for c in chosen.tolist():
        per_source_kept[sources[pool_owner[c][0]].name] = per_source_kept.get(sources[pool_owner[c][0]].name, 0) + 1
    selection["kept_per_source"] = per_source_kept
    selection["qwen_rows_all_kept"] = True
    selection["qwen_rows_kept"] = f"0..{q - 1} (every trunk row, verbatim, in order)"
    record["selection"] = selection

    # -- slice the kept columns, free the pool Gram, then the check Gram on kept ----
    g_all_k = full["gram"][kept][:, kept].clone()
    b_all_k = full["cross"][kept].clone()
    del full
    gc.collect()
    record["peak_rss_gigabytes_after_selection"] = rss_gigabytes()
    gate_kept_dev = gate_all[kept]
    up_kept_dev = up_all[kept]
    del gate_all, up_all

    def kept_features(x: torch.Tensor) -> torch.Tensor:
        return F.silu(x @ gate_kept_dev.T) * (x @ up_kept_dev.T)

    chk = accumulate_gram(kept_features, x_fit, y_fit, check & finite_rows, int(kept.numel()), device, chunk, block)
    record["gram"]["check_gram_width"] = int(kept.numel())
    record["gram"]["check_pass_nonfinite_dropped"] = chk["dropped"]
    g_chk_k, b_chk_k, y_sq_chk = chk["gram"], chk["cross"], float(chk["y_sq"])
    del chk
    # inner = all - check on the KEPT columns; law_abiding_solve_gram adds them back in place
    g_inner = g_all_k.sub_(g_chk_k)
    b_inner = b_all_k.sub_(b_chk_k)
    prior = torch.cat([trunk_down.float(), prior_p[:, chosen]], dim=1)
    try:
        solved, ladder = c14.law_abiding_solve_gram(g_inner, b_inner, g_chk_k, b_chk_k, y_sq_chk, prior)
    except SystemExit as refusal:
        raise SystemExit(f"layer {k}: {refusal}") from refusal
    except qf_solve.GramNotPositiveDefinite as defect:
        raise SystemExit(f"layer {k}: {defect}") from defect
    solved = solved.float()
    del g_inner, b_inner, g_chk_k, b_chk_k
    gc.collect()
    record["ridge"] = ladder

    # -- pad to the uniform width when the pool was short --------------------------
    width = q + k3_channels
    gate_child = torch.cat([gate_q, gate_p[chosen]])
    up_child = torch.cat([up_q, up_p[chosen]])
    down_child = solved
    pad = width - int(gate_child.shape[0])
    if pad > 0:
        gate_child = torch.cat([gate_child, torch.zeros(pad, h)])
        up_child = torch.cat([up_child, torch.zeros(pad, h)])
        down_child = torch.cat([down_child, torch.zeros(h, pad)], dim=1)
    record["installed"] = {"intermediate_size": width, "qwen_rows": q, "k3_rows": int(chosen.numel()),
                           "zero_pad_rows": pad}

    # -- cast to the FFN dtype FIRST: the receipt measures the shipped weights --------
    up_ship = torch.cat([gate_child, up_child]).to(ffn_dtype).contiguous()
    down_ship = down_child.to(ffn_dtype).contiguous()
    gate_c = up_ship[:width].float()
    up_c = up_ship[width:].float()
    down_c = down_ship.float()
    ref_prior = float(prior.norm())
    kept_n = int(kept.numel())
    record["update_scale_fp32_solution"] = float(ladder["update_scale"])
    record["update_scale"] = (float((down_c[:, :kept_n] - prior).norm() / ref_prior) if ref_prior > 0 else 0.0)
    record["cast"] = {"ffn_dtype": str(ffn_dtype).replace("torch.", ""),
                      "down_cast_relL2": relative_l2(down_c, down_child),
                      "up_cast_relL2": relative_l2(torch.cat([gate_c, up_c]), torch.cat([gate_child, up_child])),
                      "rule": "scores and update_scale are measured on the weights cast to the FFN dtype (what is "
                              "installed live and shipped); ridge.update_scale is the fp32 ladder number"}
    record["update_scale_cast_exceeds_law"] = bool(record["update_scale"] > 1.0)

    # -- scores -------------------------------------------------------------------
    scores: dict[str, Any] = {}
    with torch.no_grad():
        gk = gate_c.to(device)
        uk = up_c.to(device)
        for label, x_rows, y_rows in (("fit", x_fit, y_fit), ("held", x_held, y_held)):
            if not x_rows.shape[0]:
                scores[label] = None
                continue
            pred, no_repair, trunk_only = [], [], []
            gq_d, uq_d, dq_d = gate_q.to(device), up_q.to(device), trunk_down.float().to(device)
            dk = down_c.to(device)
            pk = torch.cat([prior, torch.zeros(h, pad)], dim=1).to(device)
            for start in range(0, int(x_rows.shape[0]), chunk):
                x = x_rows[start:start + chunk].to(device)
                a = F.silu(x @ gk.T) * (x @ uk.T)
                pred.append((a @ dk.T).cpu())
                no_repair.append((a @ pk.T).cpu())
                trunk_only.append(((F.silu(x @ gq_d.T) * (x @ uq_d.T)) @ dq_d.T).cpu())
            pred_t, nr_t, tr_t = torch.cat(pred), torch.cat(no_repair), torch.cat(trunk_only)
            scores[label] = {"union": write_scores(pred_t, y_rows), "no_repair_prior": write_scores(nr_t, y_rows),
                             "trunk_only": write_scores(tr_t, y_rows), "rows": int(x_rows.shape[0])}
    record["scores"] = scores
    record["scores_dtype"] = str(ffn_dtype).replace("torch.", "")
    held_rel = scores["held"]["union"]["relL2"] if scores.get("held") else None
    record["heldout_relL2"] = held_rel

    # -- the veto, out loud ---------------------------------------------------------
    absent = (held_energy is None and fit_energy < K3_ENERGY_FLOOR) or (
        held_energy is not None and held_energy < K3_ENERGY_FLOOR) or bool(selection.get("rank_starved"))
    ridge_veto = bool(ladder.get("veto_binding"))
    reasons = []
    if ridge_veto:
        reasons.append("THE LAW: the free solve wanted more than the prior (ridge.veto_binding)")
    if absent:
        reasons.append(f"K3 rows present but no K3 function: held-out K3-term energy fraction "
                       f"{held_energy if held_energy is not None else fit_energy!r} < {K3_ENERGY_FLOOR:g}"
                       + (" and the pool Gram was rank-starved" if selection.get("rank_starved") else ""))
    record["k3_rows_function_absent"] = bool(absent)
    record["veto_binding"] = bool(ridge_veto or absent)
    record["veto_reason"] = "; ".join(reasons) if reasons else "not binding"
    record["seconds"] = round(time.time() - started, 2)
    record["peak_rss_gigabytes"] = rss_gigabytes()

    if dump_dir:
        from safetensors.torch import save_file

        os.makedirs(dump_dir, exist_ok=True)
        save_file({"x_fit": x_fit.contiguous(), "x_held": x_held.contiguous(), "y_fit": y_fit.contiguous(),
                   "y_held": y_held.contiguous(), "k3_term_fit": term_fit.contiguous(),
                   "k3_term_held": term_held.contiguous(), "check_rows": check.contiguous(),
                   "kept_pool_index": chosen.contiguous(), "pool_gate": gate_p.contiguous(),
                   "pool_up_scaled": up_p.contiguous(), "pool_prior": prior_p.contiguous()},
                  os.path.join(dump_dir, f"L{k:02d}.safetensors"), metadata={"layer": str(k)})
        record["captures"] = os.path.join(dump_dir, f"L{k:02d}.safetensors")

    return LayerResult(up_ship, down_ship, record)


# --------------------------------------------------------------------------
# per-layer persistence and resume
# --------------------------------------------------------------------------
class Partial:
    """``<out>.partial/``: L{k:02d}.safetensors + L{k:02d}.json per completed layer, sealed by
    ``partial_inputs.json`` -- beside the child, never inside it (oc_seal counts every
    *.safetensors under a checkpoint root)."""

    def __init__(self, out: Path) -> None:
        self.dir = Path(f"{out}.partial")

    @property
    def inputs_path(self) -> Path:
        return self.dir / PARTIAL_INPUTS

    def layer_files(self, k: int) -> tuple[Path, Path]:
        return self.dir / f"L{k:02d}.safetensors", self.dir / f"L{k:02d}.json"

    def start(self, identity: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        write_receipt(str(self.inputs_path), {"schema": PARTIAL_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
                                              "identity": identity})

    def verify_identity(self, identity: dict[str, Any]) -> None:
        if not self.inputs_path.is_file():
            raise SystemExit(f"--resume: {self.inputs_path} does not exist; nothing to resume")
        doc = json.loads(self.inputs_path.read_text(encoding="utf-8"))
        if doc.get("schema") != PARTIAL_SCHEMA:
            raise SystemExit(f"--resume: {self.inputs_path} schema {doc.get('schema')!r} != {PARTIAL_SCHEMA!r}")
        stored = doc.get("identity", {})
        differing = sorted(key for key in set(stored) | set(identity) if stored.get(key) != identity.get(key))
        if differing:
            raise SystemExit(f"--resume: the inputs differ from the partial run's on {differing}; a resume must "
                             f"continue the SAME install (clean rebuild: --overwrite without --resume)")

    def completed(self, n_layers: int) -> list[int]:
        """The contiguous prefix 0..m of persisted layers (a gap ends it)."""
        done = []
        for k in range(n_layers):
            tensors, record = self.layer_files(k)
            if tensors.is_file() and record.is_file():
                done.append(k)
            else:
                break
        return done

    def save(self, k: int, result: LayerResult) -> dict[str, str]:
        from safetensors.torch import save_file

        tensors, record_path = self.layer_files(k)
        shas = {"up_sha256": sha256_tensor_bytes(result.up_weight), "down_sha256": sha256_tensor_bytes(result.down_weight)}
        result.record["installed"].update(shas)
        save_file({"up": result.up_weight.contiguous(), "down": result.down_weight.contiguous()}, str(tensors),
                  metadata={"layer": str(k), "tool": TOOL_NAME, "tool_version": TOOL_VERSION})
        result.record["partial"] = {"tensors": str(tensors), "tensors_sha256": sha256_file(str(tensors))}
        write_receipt(str(record_path), result.record)
        return shas

    def load(self, k: int, chart: Chart | None, k3_files: dict[int, str], ffn_dtype: torch.dtype,
             width: int, h: int) -> LayerResult:
        from safetensors.torch import load_file

        tensors, record_path = self.layer_files(k)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if int(record.get("layer", -1)) != k:
            raise SystemExit(f"--resume: {record_path} is for layer {record.get('layer')}, not {k}")
        chart_rec = record.get("chart")
        if (chart is None) != (chart_rec is None) or (chart is not None and chart_rec["sha256"] != chart.sha256):
            raise SystemExit(f"--resume: layer {k} was solved with chart {chart_rec and chart_rec['sha256']}, "
                             f"the current chart is {chart and chart.sha256}")
        for src in record.get("k3_sources", []):
            l = int(src["k3_layer"])
            actual = sha256_file(k3_files[l]) if l in k3_files else None
            if actual != src["sha256"]:
                raise SystemExit(f"--resume: layer {k} consumed K3 L{l:02d} {src['sha256']}, the current file "
                                 f"hashes {actual}")
        if [int(x) for x in record["k3_layers"]] != sorted(k3_files):
            raise SystemExit(f"--resume: layer {k} pooled K3 layers {record['k3_layers']}, the alignment now says "
                             f"{sorted(k3_files)}")
        got = load_file(str(tensors))
        up_weight, down_weight = got["up"], got["down"]
        if up_weight.dtype != ffn_dtype or down_weight.dtype != ffn_dtype:
            raise SystemExit(f"--resume: {tensors} carries {up_weight.dtype}/{down_weight.dtype}, the FFN dtype is "
                             f"{ffn_dtype}")
        if tuple(up_weight.shape) != (2 * width, h) or tuple(down_weight.shape) != (h, width):
            raise SystemExit(f"--resume: {tensors} shapes {tuple(up_weight.shape)}/{tuple(down_weight.shape)} do not "
                             f"match width {width} x hidden {h}")
        installed = record["installed"]
        if sha256_tensor_bytes(up_weight) != installed["up_sha256"] or sha256_tensor_bytes(down_weight) != installed["down_sha256"]:
            raise SystemExit(f"--resume: {tensors} does not hash like its record")
        record["resumed_from_partial"] = True
        return LayerResult(up_weight, down_weight, record)

    def remove(self) -> None:
        if self.dir.is_dir() and self.inputs_path.is_file():
            shutil.rmtree(self.dir)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trunk", required=True, help="the S3 Chimera trunk (uniform C2 width, SiLU)")
    ap.add_argument("--parent-trunk", required=True, help="the S0 Chimera import of the parent (targets)")
    ap.add_argument("--k3-dense", required=True, help="S2 output: L00/mlp.safetensors + L{l}/ffn.safetensors")
    ap.add_argument("--k3-form", default="E", choices=("E",), help="only form E is installed (form M is export-only)")
    ap.add_argument("--k3-layers", default=None, help="subset like 0-6 or 0,1,2 (default: every layer present)")
    ap.add_argument("--k3-config", default=None, help="K3 config.json (rms_norm_eps, activation_situ_*); REQUIRED "
                                                       "unless --k3-config-from-flags")
    ap.add_argument("--k3-config-from-flags", action="store_true", help="tests: take the K3 constants from the flags")
    ap.add_argument("--k3-rms-eps", type=float, default=1e-5)
    ap.add_argument("--k3-situ-beta", type=float, default=4.0)
    ap.add_argument("--k3-situ-linear-beta", type=float, default=25.0, help="<= 0 means uncapped")
    ap.add_argument("--charts", default=None, help="S4 charts as c17 writes them: L{k:02d}/<arm>.safetensors + "
                                                   "layer.json, alignment.json")
    ap.add_argument("--chart-arm", default="ridge", choices=tuple(CHART_ARM_FILES),
                    help="which c17 arm to install through (ridge = the chart of record)")
    ap.add_argument("--chart-bias-ceiling", type=float, default=0.1,
                    help="refuse when a dropped input-bias shift or the excluded output bias exceeds this relL2 "
                         "(0 disables the refusal; the numbers are recorded either way)")
    ap.add_argument("--alignment", default=None, help="alignment.json override (c17 table or {trunk_to_k3: {k: [l, ...]}})")
    ap.add_argument("--identity-charts", action="store_true", help="tests only: identity charts when widths agree")
    ap.add_argument("--synthetic-fixture-root", help="explicit marked CPU-only tiny unit-test workspace; never production")
    ap.add_argument("--evidence", required=True, help="the sealed omni-evidence manifest (qf_corpus schema)")
    ap.add_argument("--k3-channels", type=int, default=12288, help="K3 rows per layer; total width = trunk + this")
    ap.add_argument("--extra-blocks", action="append", default=[], help="name=path@trunk_layer (S7 blend arm)")
    ap.add_argument("--extra-charts", action="append", default=[], help="name=chart.safetensors for --extra-blocks")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help="model dtype (default: the checkpoint's)")
    ap.add_argument("--engine", default="streamed", choices=tuple(ENGINES),
                    help="streamed (host residual, one block on the device) or resident (whole model; tests)")
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--gram-block", type=int, default=16384)
    ap.add_argument("--parent-parameter-count", type=int, default=None)
    ap.add_argument("--parent-record", default=None, help="ledger record JSON carrying parameter_count")
    ap.add_argument("--out", required=True, help="child directory (under /mnt/lexar/omni-campaign/composite/ on the box)")
    ap.add_argument("--receipt", default=None, help="extra copy of union_receipt.json (small; $OC/composite/s5/)")
    ap.add_argument("--dump-captures", action="store_true", help="write per-layer captures beside the receipt (tests)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.partial after verifying the inputs")
    ap.add_argument("--stop-after-layer", type=int, default=None,
                    help=f"persist through this layer, then exit {EXIT_STOPPED} (detached runs; tests)")
    ap.add_argument("--overwrite", action="store_true", help="clean rebuild: remove --out (must carry this tool's marker)")
    return ap


def parse_layer_spec(text: str | None, present: list[int]) -> list[int]:
    if not text:
        return list(present)
    wanted: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.extend(range(int(lo), int(hi) + 1))
        elif part:
            wanted.append(int(part))
    missing = [l for l in wanted if l not in present]
    if missing:
        raise SystemExit(f"--k3-layers names layers not present under --k3-dense: {missing}")
    return sorted(set(wanted))


def verbatim_copy(trunk: Path, out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for path in checkpoint_files(trunk):
        shutil.copyfile(path, out / path.name)
        digests[path.name] = sha256_file(str(out / path.name))
    return digests


def _extra_paths(args: argparse.Namespace) -> list[tuple[str, str]]:
    paths = []
    for spec in args.extra_blocks:
        if "=" in spec and "@" in spec:
            paths.append(("--extra-blocks", spec.split("=", 1)[1].rsplit("@", 1)[0]))
    for spec in args.extra_charts:
        if "=" in spec:
            paths.append(("--extra-charts", spec.split("=", 1)[1]))
    return paths


def guard_out(args: argparse.Namespace, out: Path) -> Path:
    """``--out`` may not overlap ANY input path, and ``--overwrite`` removes only what this tool made."""
    inputs = [("--trunk", args.trunk), ("--parent-trunk", args.parent_trunk), ("--k3-dense", args.k3_dense),
              ("--charts", args.charts), ("--evidence", args.evidence), ("--receipt", args.receipt),
              ("--alignment", args.alignment), ("--k3-config", args.k3_config), ("--parent-record", args.parent_record)]
    inputs.extend(_extra_paths(args))
    partial = Path(f"{out}.partial")
    for label, raw in inputs:
        if not raw:
            continue
        src = Path(raw).absolute()
        for target, what in ((out, "--out"), (partial, "<out>.partial")):
            if target == src or target in src.parents or src in target.parents:
                raise SystemExit(f"{what} {target} overlaps {label} {src}")
    stale = Partial(out)
    if not args.resume and stale.inputs_path.is_file():
        if not args.overwrite:
            raise SystemExit(f"{stale.dir} holds a partial run; pass --resume to continue it or --overwrite to "
                             f"discard it")
        stale.remove()
    if out.exists() and any(out.iterdir()):
        entries = sorted(p.name for p in out.iterdir())
        if args.resume:
            if entries != [OUT_MARKER]:
                raise SystemExit(f"--resume: {out} holds {entries}; the child is written only at the end -- remove "
                                 f"the stale child yourself if it is one")
        elif not args.overwrite:
            raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
        elif not ((out / OUT_MARKER).is_file() or (out / "union_receipt.json").is_file()):
            raise SystemExit(f"--overwrite refuses to remove {out}: it carries neither {OUT_MARKER} nor "
                             f"union_receipt.json, so this tool did not make it")
        else:
            shutil.rmtree(out)
    return partial


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    device = canonical_device(args.device)
    out = Path(args.out).absolute()
    trunk_dir = Path(args.trunk).absolute()
    parent_dir = Path(args.parent_trunk).absolute()
    if args.k3_channels < 0:
        raise SystemExit("--k3-channels must be >= 0")
    if args.group_size < 1:
        raise SystemExit("--group-size must be >= 1")

    trunk_doc = read_checkpoint_config(trunk_dir)
    trunk_cfg = ChimeraConfig.from_manifest(trunk_doc["config"])
    parent_doc = read_checkpoint_config(parent_dir)
    parent_cfg = ChimeraConfig.from_manifest(parent_doc["config"])
    if trunk_cfg.activation_kind != "silu":
        raise SystemExit(f"the trunk must be a SiLU-GLU child (activation_kind {trunk_cfg.activation_kind!r}); "
                         "the K3 rows are rescaled INTO SiLU, not the other way")
    if trunk_cfg.hidden_size != parent_cfg.hidden_size or trunk_cfg.num_layers != parent_cfg.num_layers:
        raise SystemExit("trunk and parent trunk disagree on hidden_size / num_layers")
    trunk_widths = {trunk_cfg.resolved_intermediate_size(k) for k in range(trunk_cfg.num_layers)}
    if len(trunk_widths) != 1:
        raise SystemExit(f"the trunk is not uniform: intermediate widths {sorted(trunk_widths)}")
    q = trunk_widths.pop()
    width = q + args.k3_channels
    n_layers = int(trunk_cfg.num_layers)
    h = int(trunk_cfg.hidden_size)
    fixture = fixture_root_checked(args.synthetic_fixture_root,
        [args.trunk, args.parent_trunk, args.k3_dense, args.charts, args.evidence, args.out,
         *[p for _, p in _extra_paths(args)]], hidden=h, source_hidden=h, device=str(device), layers=n_layers)
    if fixture and any(d.get("source", {}).get("import_source") != "c10 unit test" for d in (trunk_doc, parent_doc)):
        raise SystemExit("synthetic fixture route requires explicit unit-test parent/trunk source markers")
    if not fixture and (args.identity_charts or args.extra_blocks or args.extra_charts):
        raise SystemExit("identity/extra-block charts lack native read/write boundary evidence; production route refused")
    if not fixture and any(any(tag in json.dumps(d.get("source", {})).lower()
                               for tag in ("synthetic", "fixture", "unit test")) for d in (trunk_doc, parent_doc)):
        raise SystemExit("synthetic checkpoint source cannot enter production; explicit tiny fixture root is required")

    parent_count, parent_count_record = resolve_parent_parameter_count(args)
    trunk_total, trunk_counts = header_total(trunk_dir)
    projected = trunk_total + n_layers * 3 * h * args.k3_channels
    trunk_digests = directory_file_digests(trunk_dir)
    parent_digests = directory_file_digests(parent_dir)
    receipt: dict[str, Any] = {
        "schema": "neural-forge-c10-union-install-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "tool_sha256": sha256_file(os.path.abspath(__file__)), "argv": list(argv) if argv is not None else sys.argv[1:],
        "torch": torch.__version__, "device": str(device), "engine": args.engine,
        "synthetic_fixture": bool(fixture), "fixture_root": fixture,
        "lanes": {"k3": K3_LANE_DIR, "qwen_fold": QF_LANE_DIR, "ledger": LEDGER_DIR, "chimera_src": CHIMERA_SRC},
        "trunk": {"path": str(trunk_dir), "files": trunk_digests, "header_parameters": trunk_total,
                  "intermediate_size": q, "num_layers": n_layers, "hidden_size": h,
                  "chimera_commit": trunk_doc.get("chimera_commit")},
        "parent_trunk": {"path": str(parent_dir), "files": parent_digests,
                         "activation_kind": parent_cfg.activation_kind,
                         "intermediate_size": int(parent_cfg.intermediate_size)},
        "parent_parameter_count": parent_count, "parent_parameter_count_source": parent_count_record,
        "k3_channels": args.k3_channels, "intermediate_size": width, "k3_form": args.k3_form,
        "size_gate": {"projected_child_header_parameters": projected, "rule": "child < parent record parameter_count"},
        "layers": [],
    }
    refuse_size(projected, parent_count, "before any capture, from the arithmetic")

    # -- k3-channels 0: the trunk, byte for byte --------------------------------
    if args.k3_channels == 0:
        guard_out(args, out)
        digests = verbatim_copy(trunk_dir, out)
        child_total, _ = header_total(out)
        refuse_size(child_total, parent_count, "after the verbatim copy", out)
        receipt.update({"mode": "verbatim copy of the trunk (--k3-channels 0)", "child": {"path": str(out),
                        "files": digests, "header_parameters": child_total},
                        "byte_identical_to_trunk": digests == receipt["trunk"]["files"],
                        "veto_binding_layers": 0, "seconds": round(time.time() - started, 2)})
        if not receipt["byte_identical_to_trunk"]:
            raise SystemExit("the verbatim copy does not hash like the trunk")
        receipt["receipt_sha256"] = write_receipt(str(out / "union_receipt.json"), receipt)
        if args.receipt:
            shutil.copyfile(out / "union_receipt.json", args.receipt)
        log(f"child == trunk byte for byte; receipt -> {out / 'union_receipt.json'}")
        return 0

    # -- inputs ----------------------------------------------------------------------
    k3_cfg, k3_cfg_record = load_k3_config(args.k3_config, args)
    present = discover_k3_layers(args.k3_dense)
    wanted = parse_layer_spec(args.k3_layers, present)
    k3_headers = {l: k3_layer_header(args.k3_dense, l) for l in wanted}      # headers only: lazy loading
    k3_hidden = {hdr["hidden"] for hdr in k3_headers.values()}
    if len(k3_hidden) != 1:
        raise SystemExit(f"K3 layers disagree on hidden size: {sorted(k3_hidden)}")
    k3_hidden_size = k3_hidden.pop()
    fixture_root_checked(fixture, [args.k3_dense], hidden=h, source_hidden=k3_hidden_size,
                         device=str(device), layers=n_layers)
    alignment, alignment_record = load_alignment(args.charts, args.alignment, wanted, n_layers)
    charts: dict[int, Chart] = {}
    for k in range(n_layers):
        if alignment[k]:
            charts[k] = load_chart(args.charts, k, k3_hidden_size, h, args.identity_charts, args.chart_arm,
                                   fixture_root=fixture)
            if not fixture:
                provenance = charts[k].native_boundary_provenance
                if (provenance["host_index_sha256"] != sha256_file(str(trunk_dir / INDEX_FILENAME))
                        or provenance["evidence_sha256"] != sha256_file(args.evidence)
                        or provenance["source_layers"] != sorted(alignment[k])):
                    raise SystemExit("native chart boundaries do not match installed host/evidence/source alignment")
                for l in alignment[k]:
                    if provenance["installed_ffn_sha256"][str(l)] != sha256_file(k3_headers[l]["path"]):
                        raise SystemExit("native chart source capture does not match installed K3 FFN")
    extra_charts = {}
    for spec in args.extra_charts:
        if "=" not in spec:
            raise SystemExit(f"--extra-charts expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        extra_charts[name] = path
    extras: dict[int, list[Source]] = {}
    extra_records = []
    for spec in args.extra_blocks:
        k, source, rec = load_extra_block(spec, extra_charts, h)
        extras.setdefault(k, []).append(source)
        extra_records.append(rec)
    evidence = load_evidence(args.evidence)
    receipt.update({
        "k3": {"path": os.path.abspath(args.k3_dense), "layers": wanted, "hidden": k3_hidden_size,
               "config": k3_cfg_record, "loading": "lazy: alignment[k] per trunk layer, released after the install",
               "files": {f"L{l:02d}": {"path": hdr["path"], "kind": hdr["kind"], "routed_width": hdr["routed_width"],
                                       "shared_width": hdr["shared_width"], "dense_width": hdr["dense_width"]}
                         for l, hdr in k3_headers.items()}},
        "alignment": alignment_record,
        "charts": {"dir": os.path.abspath(args.charts) if args.charts else "identity", "arm": args.chart_arm,
                   "alignment_json_sha256": alignment_record.get("sha256"),
                   "bias_ceiling": args.chart_bias_ceiling,
                   "layers": {str(k): c.record() for k, c in charts.items()}},
        "extra_blocks": extra_records,
        "evidence": {"path": evidence.path, "file_sha256": evidence.sha256, "manifest_sha256": evidence.manifest_sha256,
                     "fit_batches": evidence.fit_batches, "held_batches": evidence.held_batches,
                     "sequence_length": evidence.sequence_length, "check_rows": int(evidence.check_rows.sum())},
        "group_size": int(args.group_size),
        "schedule": "groups of --group-size trunk layers: one capture pass per group against the LIVE model, "
                    "which already carries every previous group's union FFN (qf_mlp ordering law)",
    })

    # -- the trunk engine ------------------------------------------------------------------
    guard_out(args, out)
    load_dtype = getattr(torch, args.dtype) if args.dtype else None
    trunk_views = read_safetensors(trunk_dir, verify=True)
    engine = ENGINES[args.engine](trunk_dir, trunk_cfg, trunk_views, device, load_dtype, evidence)
    ffn_dtype = engine.ffn_dtype
    receipt["trunk"]["load"] = engine.record()
    receipt["trunk"]["zero_gains"] = [zero_gain_census(n, v.torch(device="cpu", dtype=torch.float32))
                                      for n, v in trunk_views.items() if n.endswith("norm2.weight")
                                      and bool((v.torch(device="cpu", dtype=torch.float32) == 0).any())]

    # -- persistence: identity + resume ------------------------------------------------------
    identity = {
        "trunk": trunk_digests, "parent_trunk": parent_digests,
        "evidence": {"file_sha256": evidence.sha256, "manifest_sha256": evidence.manifest_sha256},
        "k3_dense": os.path.abspath(args.k3_dense), "k3_layers": wanted, "k3_config": k3_cfg_record,
        "charts": {str(k): c.sha256 for k, c in charts.items()}, "chart_arm": args.chart_arm,
        "chart_boundary_provenance": {str(k): c.native_boundary_provenance for k, c in charts.items()},
        "chart_layer_json_sha256": {str(k): c.layer_json_sha256 for k, c in charts.items()},
        "synthetic_fixture": bool(fixture),
        "alignment": alignment_record.get("trunk_to_k3"), "alignment_sha256": alignment_record.get("sha256"),
        "extra_blocks": extra_records, "k3_channels": int(args.k3_channels), "group_size": int(args.group_size),
        "engine": args.engine, "ffn_dtype": str(ffn_dtype).replace("torch.", ""),
        "chart_bias_ceiling": float(args.chart_bias_ceiling), "chunk": int(args.chunk), "gram_block": int(args.gram_block),
        "tool_version": TOOL_VERSION,
    }
    partial = Partial(out)
    completed: list[int] = []
    if args.resume:
        partial.verify_identity(identity)
        completed = partial.completed(n_layers)
        log(f"resume: {len(completed)} completed layer(s) {completed[:3]}..{completed[-3:] if completed else []}; "
            f"continuing from layer {len(completed)}")
    else:
        partial.start(identity)
    out.mkdir(parents=True, exist_ok=True)
    write_receipt(str(out / OUT_MARKER), {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "started": started,
                                          "partial": str(partial.dir)})
    receipt["partial"] = {"dir": str(partial.dir), "resumed": bool(args.resume), "layers_reloaded": list(completed)}

    new_state: dict[str, Any] = {}
    groups = [list(range(s, min(s + args.group_size, n_layers))) for s in range(0, n_layers, args.group_size)]
    # captures live BESIDE the child, never inside it: oc_seal walks every
    # *.safetensors under the checkpoint root and would count them as parameters
    dump_dir = f"{out}.captures" if args.dump_captures else None
    if dump_dir:
        receipt["captures_dir"] = dump_dir
    veto_layers = 0
    stopped = False
    for group in groups:
        todo = [k for k in group if k not in completed]
        captured = None
        if todo:
            log(f"capturing group {group} on {device} ({engine.name}; solving {todo})")
            captured = engine.capture(group)
        for k in group:
            trunk_up = trunk_views[f"blocks.{k}.ffn.up.weight"].torch(device="cpu", dtype=torch.float32)
            trunk_down = trunk_views[f"blocks.{k}.ffn.down.weight"].torch(device="cpu", dtype=torch.float32)
            k3_files = {l: k3_headers[l]["path"] for l in alignment[k]}
            if k in completed:
                result = partial.load(k, charts.get(k), k3_files, ffn_dtype, width, h)
                log(f"  L{k:02d}: reloaded from {partial.dir}")
            else:
                assert captured is not None
                parent = load_parent_mlp(str(parent_dir), k, parent_cfg, device)
                layers_here = [load_k3_layer(args.k3_dense, l, k3_cfg) for l in alignment[k]]   # lazy
                result = _run_layer(k, captured[k], trunk_up, trunk_down, trunk_cfg, parent, layers_here, charts.get(k),
                                    extras.get(k, []), evidence, args.k3_channels, device, args.chunk, args.gram_block,
                                    dump_dir, ffn_dtype, args.chart_bias_ceiling)
                for layer in layers_here:
                    layer.release()
                del layers_here
                result.record["parent_mlp_sha256"] = parent.sha256
                partial.save(k, result)
                del parent
                captured[k] = None
            receipt["layers"].append(result.record)
            veto_layers += int(result.record["veto_binding"])
            # install into the LIVE model so the next group is captured on the folded base
            engine.install(k, result.up_weight, result.down_weight)
            new_state[f"blocks.{k}.ffn.up.weight"] = result.up_weight.to(ffn_dtype).contiguous()
            new_state[f"blocks.{k}.ffn.down.weight"] = result.down_weight.to(ffn_dtype).contiguous()
            log(f"  L{k:02d}: heldout relL2 {result.record['heldout_relL2']} K3 energy "
                f"{result.record['k3_term_energy_fraction']['held']} update_scale {result.record['update_scale']:.3f} "
                f"veto_binding {result.record['veto_binding']} ({result.record['seconds']} s, rss "
                f"{result.record.get('peak_rss_gigabytes')} GB)")
            del result
            gc.collect()
            if args.stop_after_layer is not None and k >= args.stop_after_layer:
                stopped = True
                break
        del captured
        gc.collect()
        if stopped:
            break
        engine.advance(group)
    if stopped:
        engine.close()
        log(f"stopped after layer {args.stop_after_layer} as asked; {len(receipt['layers'])} layer(s) persisted under "
            f"{partial.dir}; continue with --resume")
        return EXIT_STOPPED
    engine.close()

    # -- write the child ---------------------------------------------------------------
    state: dict[str, Any] = {}
    for name, view in trunk_views.items():
        state[name] = new_state.get(name, view)
    child_cfg = dataclasses.replace(
        trunk_cfg, intermediate_size=width, intermediate_sizes=tuple([width] * n_layers),
        metadata={**trunk_cfg.metadata, "c10_union_install": f"{TOOL_NAME} {TOOL_VERSION}",
                  "c10_k3_channels": str(args.k3_channels), "c10_trunk_intermediate_size": str(q)})
    child_cfg.validate()
    source_record = {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "trunk": str(trunk_dir),
                     "parent_trunk": str(parent_dir), "k3_dense": os.path.abspath(args.k3_dense),
                     "charts": os.path.abspath(args.charts) if args.charts else "identity", "chart_arm": args.chart_arm,
                     "evidence_manifest_sha256": evidence.manifest_sha256, "k3_channels": str(args.k3_channels),
                     "trunk_source": trunk_doc.get("source")}
    (out / OUT_MARKER).unlink(missing_ok=True)
    index = write_safetensors(state, out, config=child_cfg.to_manifest(), ledger=trunk_doc.get("ledger"),
                              source=source_record, chimera_commit=trunk_doc.get("chimera_commit"),
                              metadata={"tool": TOOL_NAME, "tool_version": TOOL_VERSION})
    child_total, child_counts = header_total(out)
    receipt["child"] = {"path": str(out), "files": directory_file_digests(out), "header_parameters": child_total,
                        "tensors": len(index["weight_map"]), "total_size": index["metadata"]["total_size"],
                        "intermediate_sizes": list(child_cfg.intermediate_sizes)}
    if child_total != projected:
        shutil.rmtree(out)
        raise SystemExit(f"the written child counts {child_total:,} header parameters, the arithmetic said {projected:,}; "
                         f"{out} removed")
    refuse_size(child_total, parent_count, "after the write, from oc_seal.safetensors_tensor_counts", out)
    receipt["size_gate"].update({"child_header_parameters": child_total, "parent_parameter_count": parent_count,
                                 "headroom": parent_count - child_total, "passed": True})
    qwen_rows = 3 * h * q * n_layers
    k3_rows = 3 * h * args.k3_channels * n_layers
    receipt["components"] = {
        "qwen.mlp": {"child_parameters": qwen_rows, "prefix": "blocks.", "note": "the trunk's rows, verbatim gate/up"},
        "k3.ffn": {"child_parameters": k3_rows, "prefix": "blocks.",
                   "note": "rows share blocks.*.ffn with the Qwen rows; 3 x h x k3_channels x layers"},
    }
    receipt["veto_binding_layers"] = veto_layers
    receipt["veto_binding_most_layers"] = bool(veto_layers * 2 > n_layers)
    receipt["frontier_rule"] = ("veto binds in most layers: rows in name only, delete this child and record the rebuild "
                                "command (more paired rows / a smaller --k3-channels / the C13 mixer rung)"
                                if veto_layers * 2 > n_layers else "the K3 rows carry function in most layers")
    receipt["chart_bias_worst_input_shift_relL2"] = max(
        (l["chart_bias"]["input"]["worst_shift_relL2"] for l in receipt["layers"] if "chart_bias" in l), default=0.0)
    receipt["chart_bias_worst_output_energy_fraction"] = max(
        (l["chart_bias"]["output"]["energy_fraction_fit"] for l in receipt["layers"] if "chart_bias" in l), default=0.0)
    receipt["peak_rss_gigabytes"] = rss_gigabytes()
    receipt["peak_rss_gigabytes_per_layer"] = {str(l["layer"]): l.get("peak_rss_gigabytes") for l in receipt["layers"]}
    receipt["seconds"] = round(time.time() - started, 2)
    partial.remove()
    receipt["partial"]["removed_after_write"] = True
    receipt["receipt_sha256"] = write_receipt(str(out / "union_receipt.json"), receipt)
    if args.receipt:
        os.makedirs(os.path.dirname(os.path.abspath(args.receipt)) or ".", exist_ok=True)
        shutil.copyfile(out / "union_receipt.json", args.receipt)
    log(f"child -> {out} ({child_total:,} header params < parent {parent_count:,}); veto binds in "
        f"{veto_layers}/{n_layers} layers; receipt sha {receipt['receipt_sha256']}")
    return 0


def _run_layer(k: int, captured: dict[str, torch.Tensor], trunk_up: torch.Tensor, trunk_down: torch.Tensor,
               trunk_cfg: ChimeraConfig, parent: ParentMlp, layers_here: list[K3Layer], chart: Chart | None,
               extras: list[Source], evidence: Evidence, k3_channels: int, device: torch.device, chunk: int,
               block: int, dump_dir: str | None, ffn_dtype: torch.dtype = torch.float32,
               chart_bias_ceiling: float = 0.1) -> LayerResult:
    return install_layer(k, captured["fit"], captured["held"], trunk_up, trunk_down, trunk_cfg, parent, layers_here,
                         chart, extras, evidence, k3_channels, device, chunk, block, dump_dir, ffn_dtype,
                         chart_bias_ceiling)


if __name__ == "__main__":
    raise SystemExit(main())
