#!/usr/bin/env python3
"""k3_dense_pass.py -- the full-K3 streamed DUAL-stream C14 pass (run-book S2P/S2).

Two residual pairs advance through Kimi-K3 layer by layer:

* the TEACHER pair -- the original routed MoE (896 routed + 2 shared experts,
  dequantized from MXFP4 one layer at a time, experts kept PACKED on
  ``--experts-device``);
* the FOLDED pair -- the same non-expert body, but every MoE layer ``< l`` has
  its dense form-E FFN installed in place of the routed half.

At layer ``l`` the form-E child is fitted (``c14_union_coreset.CoresetFold``
stage1 / stage2 / refit / score, ``device_memory_estimate`` + ``enforce_budget``)
on the FOLDED pair's FFN input (``latent_in`` = W_down . post-attention
residual of the folded stream) against the TEACHER pair's routed accumulator
``y_lat`` -- ruling item (2): the child is fitted where it will actually run,
on its own prefix, to reproduce what the teacher wrote.  Router mass (which
ranks channel selection) comes from the teacher's router on the teacher's
input.  Layer 0 (dense KimiMLP) is copied VERBATIM and its tensors are
sha-matched to ``k3_layer0_import_receipt.json`` (dtype + shape + bytes, the
import lane's own recipe).

Outputs (big artefacts under ``--out``, default the Lexar SSD; small receipts
under ``--receipt-dir``):

    <out>/dense/L00/mlp.safetensors            verbatim dense MLP (+ done.json)
    <out>/dense/L{l:02d}/ffn.safetensors       form E: w1_s / w3_s / w2p / channels +
                                               verbatim latent down/up, routed_expert_norm,
                                               both shared experts (+ done.json = the
                                               per-layer DONE MARKER --resume honours)
    <out>/resid/L{l:02d}/held_rows.safetensors teacher + folded FFN-input rows of the
                                               held tokens (``--held-rows``)
    <out>/snapshots/before_L{i:02d}.safetensors + before_L{i:02d}_folded.safetensors
                                               BOTH residual pairs every --snapshot-every
                                               layers, stamped with the corpus sha
                                               (a snapshot of foreign input_ids is refused)
    <out>/review/*.safetensors                 teacher-forced logit summaries on the
                                               masked review rows (the K3 R-proxy baseline)
    <receipt-dir>/k3_dense_pass_receipt.json   (+ .partial.json after every layer)
                                               per layer: shard MB/s, dequant s, pivot s,
                                               fit relL2 (held), veto_binding, update_scale,
                                               folded-vs-teacher residual relL2 (cumulative),
                                               peak RSS; end-of-stack top-1 / KL on held
                                               tokens; staged-copy deletion log;
                                               ``determinism_sha256`` over the
                                               timing-free fields (what t11c compares)

Reuse: ``k3_stream.StreamRun`` (embed / head / model-level reads; extended here
to two pairs), ``k3_layer.StreamedLayer`` (attention halves, dense swap-in
arithmetic), ``c14_union_coreset`` (the fold, the budgets, the accounting),
``k3_common`` (shard reader, MXFP4 dequant, ``K3Index.shards_for``, the
token-chunked ``apply_attn_res``), ``k3_receipt`` (the optional layer-1 mixer
diagnostic), ``k3_review_stream.kl_top_agreement``, and chimera-omni's
``importing.kimi_k3`` (``k3_layer_tensor_names`` for the layer-0 listing,
``dequant_mxfp4`` as the cross-check on one expert tensor) when ``--chimera-src``
is importable.  The k3 lane files are imported by path and never edited.

Deviations from the run-book row, on purpose and recorded in the receipt:

* ``--src-dir`` is where the WEIGHTS are read (the Lexar copy), as in
  ``qf_stream27``; ``--model-dir`` is the recorded identity (defaults to the
  same directory).  The modeling source for the mixer diagnostic is
  ``--modeling-src`` (the run book spelt that ``--src-dir``).
* Share reads: the ``share_rate`` receipt (>= 1 MB/s) is required ONLY when the
  read source is under ``/mnt/ai_data``; ``--no-share-read`` is the DEFAULT
  whenever it is not, and asking for it while the source IS on the share is a
  contradiction that is refused.
* The pass fits on the corpus' own in-band ``fit_mask`` / ``held_mask`` (the
  sealed 18 fit / 6 held rows), not on ``CoresetFold``'s positional holdout;
  the ridge's internal check split is dealt over the fit positions in
  contiguous blocks.
* A window that starts above layer 0 with no prior snapshot is accepted when
  every earlier layer is dense (K3: layer 0 only): the pair is rebuilt from the
  embedding through those layers (recorded as ``start.auto_prefix_layers``), so
  the S2P ``--layers 1-1`` command runs as written; ``--layers 0-1`` is the
  same run spelt out.
* Model-level tensors (embedding, head) are read from ``--src-dir`` directly,
  never staged: they belong to no layer's shard plan and would otherwise hold
  a window slot for the whole pass.  A staged copy that can never be admitted
  (window full, nothing to release) or that stops making byte progress for
  ``--stage-stall-seconds`` is a NAMED refusal, not a hang.
* ``--layer0-import-receipt`` defaults to the import lane's receipt and its
  absence is a refusal; ``--no-layer0-receipt`` skips the sha-match out loud.
* ``--compute-dtype auto`` casts the F32 rows of K3's mixed checkpoint
  (conv kernels, ``A_log``, ``dt_bias``, ``o_norm``, router bias) to the
  parent's projection dtype (bf16) -- what ``transformers`` does at
  ``torch_dtype=bfloat16``; ``resolve_compute_dtype`` records the source per
  layer under ``read.compute_dtype_source``.  Verbatim writes keep the on-disk bytes.
* Non-finite fit inputs (folded ``latent_in``, teacher ``y_lat``) or a
  non-finite prefix leaving a layer are refusals (section 3b), recorded under
  ``refusal`` in the partial receipt.
* v1.2.0 -- THE MEASURED ACCEPTANCE RULE replaces THE LAW's norm-ratio veto.
  ``c14.law_abiding_solve_gram`` admitted a rung only if
  ``||W2' - prior||_F / ||prior||_F <= 1``; on a routed-expert union the prior
  is the parent column scaled by the channel's MEAN router mass (~1e-3 on
  896-expert / top-16 K3, ||prior||_F 0.427 vs ||W2 bank||_F 207.8 at L01), so
  the ratio is unbounded by construction and the veto fired at every rung
  (free 81.642, rel=100 still 1.112) BEFORE any held-out number was printed
  (``run_rung`` refits, then scores).  ``measured_solve_gram`` keeps the same
  ladder and the same ``prior_anchored_gram`` arithmetic, chooses lambda by
  argmin check-split SSE over the FULL ladder, and records the law's verdict
  (``law_verdict.would_refuse``) without letting it choose or veto.  A layer
  then installs only if ``evaluate_acceptance`` passes on the held rows the
  chooser never sees: G1 signal (held y_lat and routed-write relL2 beat the
  zero and prior baselines by ``G1_signal_margin``, in fp32 AND in the runtime
  dtype), G2 no memorisation (held / fit relL2 <= ``G2_memorisation_ratio_max``),
  G3 downstream (folded-vs-teacher prefix relL2 <= ``G3_prefix_margin`` x the
  zero-child prefix relL2 measured on the same rows, card and dtype), and G4
  at the end of the stack (held KL per fitted layer / top-1 vs the first
  measured window's calibration).  Every value, threshold and pass/fail is in
  the layer entry, the rung receipt and ``done.json``; a refused layer removes
  its own ``ffn.safetensors``, leaves no done marker and records the failing
  gate with its numbers under ``refusal``.  ``--acceptance record`` computes
  and records the verdict without enforcing it (recorded in the receipt;
  never the default).  The lane file ``c14_union_coreset.py`` is untouched.
* ``k3_dense_pass_receipt.json`` cannot carry its own sha256: it is in
  ``k3_dense_pass_receipt.sha256`` and in the ``.partial.json`` written last;
  ``--resume`` carries the NEWER of the two receipts by mtime, and only the
  layer entries whose done marker validates at this corpus / rung / sha.
* Snapshots are stamped with the rung, the tool version and the sha256 of
  every installed ``ffn.safetensors``; a snapshot whose folds no longer
  validate on disk, or that was fitted at another rung, is not a resume point.
* v1.2.1 -- every MoE marker and its layer/rung receipt carry the complete
  upstream fold lineage. Resume reuses a fit only on that same installed
  prefix; missing (legacy) lineage forces a refit. Snapshots and carried
  receipts validate the whole chain. Gate-mode resume also refuses to carry
  a record-only refusal or an unverifiable acceptance verdict.
* v1.2.2 -- native path containment and exact-byte receipts work on Windows.
  Process-lifetime peak RSS is measured using the platform counter; an
  unavailable/nonpositive reading is a named refusal, recorded as null in
  ``resources.host_peak_rss``. This is an observability requirement, not a new
  numeric budget or a changed fitting/acceptance policy.
  * v1.2.3 -- reuse also requires the same source metadata, numerical runtime,
    implementation hashes and complete corpus masks/lengths. An accepted FP32
    diagnostic is not a BF16 acceptance receipt. Legacy context is never inferred.
  * v1.2.4 -- all fold scoring streams the existing token chunks through latent
    and routed-write metrics. Every row contributes; exact full cosine quantiles
    are retained on CPU and norm sums use FP64. Thresholds and gates are unchanged.

Usage (server, lab venv; the real pass runs DETACHED and is polled):

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 $PY k3_dense_pass.py \\
      --src-dir /mnt/lexar/models/moonshotai/Kimi-K3 \\
      --input-ids $OC/composite/corpus/k3_input_ids.safetensors \\
      --review-input-ids $OC/composite/corpus/k3_review_input_ids.safetensors \\
      --layers 0-92 --rung 17408 --k-pool-factor 1 --experts-device cpu \\
      --device cuda:0 --kda-backend fla --out /mnt/lexar/omni-campaign/k3 \\
      --receipt-dir $OC/composite/receipts/k3_dense_pass --resume
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import json
import os
import sys
import threading
import time
import uuid
from typing import Any

import torch
import torch.nn.functional as F

TOOL_NAME = "k3_dense_pass"
TOOL_VERSION = "1.2.4"
SHARE_ROOT = "/mnt/ai_data"
DEFAULT_SRC_DIR = "/mnt/lexar/models/moonshotai/Kimi-K3"
DEFAULT_OUT = "/mnt/lexar/omni-campaign/k3"
DEFAULT_RECEIPT_DIR = "/home/cereal/omni-campaign/composite/receipts/k3_dense_pass"
DEFAULT_SHARE_RATE_RECEIPT = "/home/cereal/omni-campaign/composite/receipts/share_rate_2026-09.json"
DEFAULT_MODELING_SRC = "/home/cereal/omni-campaign/k3/k3_src"
DEFAULT_CHIMERA_SRC = "/home/cereal/omni-campaign/import/rq27b/tree_c5e6052/src"
MIN_SHARE_MB_PER_S = 1.0
RECEIPT_SCHEMA = "omni-campaign.k3_dense_pass.v1"
STAGING_MANIFEST = ".k3_dense_pass_staging.json"
SNAPSHOT_MANIFEST = "snapshots_manifest.json"
RECEIPT_NAME = "k3_dense_pass_receipt.json"
PARTIAL_RECEIPT_NAME = "k3_dense_pass_receipt.partial.json"
RECEIPT_SHA_SIDECAR = "k3_dense_pass_receipt.sha256"
DEFAULT_LAYER0_IMPORT_RECEIPT = "/home/cereal/omni-campaign/composite/receipts/k3_layer0_import_receipt.json"
DEFAULT_DIGEST_RECEIPT = "/home/cereal/omni-campaign/k3/logs/k3_lexar_digest_verify.json"
DEFAULT_STAGE_STALL_SECONDS = 900.0
COPY_CHUNK = 64 << 20
# the per-layer fold metrics the S2 gate reads; persisted in done.json and restored on a
# resume over the marker (a reused layer must not lose them)
FOLD_METRIC_KEYS = ("fold", "pivot_seconds", "fit_timings", "veto_binding", "update_scale",
                    "fit_relL2_heldout", "routed_write_relL2_heldout", "width_installed",
                    "law_verdict", "acceptance")
PREFIX_LINEAGE_VERSION = "k3_dense_pass.prefix_lineage.v1"
RESUME_CONTEXT_VERSION = "k3_dense_pass.resume_context.v1"
KL_ROUNDOFF_POLICY = {"version": "fp32_kl_roundoff.v1",
                      "negative_absolute_tolerance": 8 * torch.finfo(torch.float32).eps,
                      "action": "preserve raw score; normalize negative FP32 roundoff to zero"}
# ---- the measured acceptance rule (v1.2.0) --------------------------------------------
# Named, versioned constants.  Changing one is a tool-version change and the receipt names the
# calibration it came from; nothing here is tuned against the held rows of a running pass.
ACCEPTANCE_VERSION = "k3_dense_pass.acceptance.v1"
ACCEPTANCE_CALIBRATION_NAME = "acceptance_calibration.json"
ACCEPTANCE_CONSTANTS = {
    # G1: held-out relL2 (y_lat, and after routed_expert_norm + up_proj) must beat the better of
    # the zero map and the prior map by this factor, in fp32 and in the runtime dtype
    "G1_signal_margin": 0.90,
    # G2: held relL2 / fit relL2 of the installed map; the rel=1e-6 free solve at L01 sat at 1.98
    "G2_memorisation_ratio_max": 1.5,
    # G3: folded-vs-teacher prefix relL2 on the held rows vs the zero-y_lat child's, same rows/card/dtype
    "G3_prefix_margin": 0.95,
    # G4: end-of-stack held KL per fitted layer / top-1 agreement vs the first measured window
    "G4_window_kl_per_layer_budget_x": 2.0,
    "G4_window_top1_floor_x": 0.5,
    # an exact fold (uniform router, full width) has held relL2 ~1e-7 on both splits; below this
    # floor G2/G3 ratios are noise over noise and the gate passes on the absolute number
    "exact_fold_floor_relL2": 1e-3,
}
ACCEPTANCE_CALIBRATED_ON = {
    "layer": "K3 L01, K=17,408, k_pool_factor 1, corpus 24 x 4,096 (fit 69,743 / check 23,247 / held 23,186)",
    "date_utc": "2026-09-07",
    "receipts": ["composite/receipts/k3_veto_diag/k3_veto_diag_L01.json (H1)",
                 "composite/receipts/k3_veto_diag/k3_veto_diag_h2_receipt.json (H2)"],
    "numbers": {
        "law_free_solve_update_scale": 81.642, "law_min_update_scale_rel_100": 1.112,
        "prior_norm": 0.4273, "w2_bank_norm": 207.76, "prior_mass_mean": 1.649e-3,
        "held_y_lat_relL2": {"zero": 1.0, "prior": 0.9775, "rel_1_chosen": 0.6437, "rel_100": 0.8062,
                             "free_rel_1e-6": 0.7503},
        "held_routed_write_relL2": {"zero": 1.0, "prior": 0.9353, "rel_1_chosen": 0.7099, "rel_100": 0.8469,
                                    "free_rel_1e-6": 0.8457},
        "fit_y_lat_relL2": {"rel_1_chosen": 0.5527, "free_rel_1e-6": 0.3781},
        "held_over_fit": {"rel_1_chosen": 1.165, "free_rel_1e-6": 1.98},
        "prefix_held_relL2": {"zero_child": 0.5107, "free_solve_installed": 0.4319},
    },
}
# per-layer tensors that are NOT the body: the experts, the whole MoE block (its latent
# maps, norm and shared experts ride in ffn.safetensors; the router is not in the child)
# and the dense MLP (mlp.safetensors)
BODY_EXCLUDED_PREFIXES = ("block_sparse_moe.", "mlp.")

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# the k3 lane, imported by path and never edited
# --------------------------------------------------------------------------
def resolve_lane_dir() -> str:
    """``$K3_LANE_DIR``, else ``<campaign>/k3/harness`` (server), else ``<campaign>/k3`` (mirror)."""
    env = os.environ.get("K3_LANE_DIR")
    candidates = [env] if env else []
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    candidates += [os.path.join(root, "k3", "harness"), os.path.join(root, "k3")]
    for cand in candidates:
        if cand and os.path.exists(os.path.join(cand, "k3_common.py")) and os.path.exists(
                os.path.join(cand, "c14_union_coreset.py")):
            return os.path.abspath(cand)
    raise SystemExit(f"the k3 lane (k3_common.py, c14_union_coreset.py, ...) was not found in {candidates}; "
                     "set K3_LANE_DIR")


LANE_DIR = resolve_lane_dir()
if LANE_DIR not in sys.path:
    sys.path.insert(0, LANE_DIR)

import c14_union_coreset as c14  # noqa: E402
from k3_common import (  # noqa: E402
    K3Index,
    apply_attn_res,
    canonical_device,
    device_inventory,
    log,
    peak_rss_bytes,
    peak_rss_source,
    read_header,
    read_metadata,
    read_tensors,
    rms_norm,
    save_tensors,
    sha256_file,
    sha256_tensor,
    situ,
    write_receipt,
    write_scores,
    WriteScoreAccumulator,
)
from k3_layer import LayerWeights, StreamedLayer, dense_mlp_forward, dequant_expert, route  # noqa: E402
from k3_review_stream import kl_top_agreement  # noqa: E402
from k3_stream import StreamRun, parse_layers  # noqa: E402


def lane_file_digests() -> dict[str, str]:
    out = {}
    for name in ("k3_common.py", "k3_layer.py", "k3_stream.py", "c14_union_coreset.py",
                 "k3_receipt.py", "k3_review_stream.py"):
        path = os.path.join(LANE_DIR, name)
        out[name] = sha256_file(path) if os.path.exists(path) else None
    return out


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def under_share(path: str | None, share_root: str = SHARE_ROOT) -> bool:
    if not path:
        return False
    # A trailing separator also makes a bare Windows UNC share an absolute root
    # to commonpath (\\\\server\\share and \\\\server\\share\\ parse differently).
    resolved = os.path.normcase(os.path.realpath(path)).rstrip(os.path.sep) + os.path.sep
    root = os.path.normcase(os.path.realpath(share_root)).rstrip(os.path.sep) + os.path.sep
    try:
        return os.path.commonpath([resolved, root]).rstrip(os.path.sep) + os.path.sep == root
    except ValueError:  # distinct Windows drives or incompatible path roots
        return False


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def import_lane_tensor_sha256(dtype_name: str, shape: list[int], buffer: bytes) -> str:
    """chimera-omni ``importing.safetensors_io._tensor_sha256``: dtype + shape + bytes."""
    digest = hashlib.sha256()
    digest.update(dtype_name.encode("ascii"))
    digest.update(json.dumps([int(x) for x in shape], separators=(",", ":")).encode("ascii"))
    digest.update(buffer)
    return digest.hexdigest()


def read_raw_tensor_records(path: str, names: list[str]) -> dict[str, dict[str, Any]]:
    """Raw on-disk bytes of the named tensors in one shard, with the import lane's sha."""
    header, base = read_header(path)
    out: dict[str, dict[str, Any]] = {}
    with open(path, "rb") as handle:
        for name in names:
            meta = header[name]
            start, stop = meta["data_offsets"]
            handle.seek(base + start)
            buf = handle.read(stop - start)
            out[name] = {
                "dtype": meta["dtype"],
                "shape": list(meta["shape"]),
                "bytes": len(buf),
                "sha256_import_lane": import_lane_tensor_sha256(meta["dtype"], meta["shape"], buf),
                "sha256_raw": sha256_bytes(buf),
            }
    return out


def zero_gain_census(name: str, weight: torch.Tensor) -> dict[str, Any]:
    """Section 3b: exact-zero RMSNorm gains exist; never divide by one, always record them."""
    zeros = torch.nonzero(weight.detach().float().cpu() == 0.0).flatten().tolist()
    return {"tensor": name, "zero_gain_coordinates": zeros[:64], "zero_gain_count": len(zeros)}


def torch_dtype(name: str) -> torch.dtype | None:
    return {"auto": None, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp32": torch.float32, "float32": torch.float32,
            "fp16": torch.float16, "float16": torch.float16}[name]


def require_finite(what: str, tensor: torch.Tensor, layer_idx: int) -> dict[str, Any]:
    """Section 3b: a non-finite stream / Gram input is a REFUSAL, named, never a NaN relL2 in a log."""
    finite = torch.isfinite(tensor)
    bad = int(tensor.numel() - int(finite.sum()))
    if bad:
        rows = int((~finite).flatten(1).any(dim=1).sum()) if tensor.dim() > 1 else bad
        raise SystemExit(f"layer {layer_idx}: {what} carries {bad} non-finite values over {rows} rows "
                         f"(shape {list(tensor.shape)}, dtype {tensor.dtype}); section 3b: a non-finite Gram input "
                         "or residual stream is a refusal, not a number to paper over")
    return {"tensor": what, "elements": int(tensor.numel()), "non_finite": 0}


def digest_receipt_status(got: dict[str, Any], path: str, expected_files: int) -> dict[str, Any]:
    """`k3_lexar_digest_verify.json` is written INCREMENTALLY: ``all_verified`` means every
    one of the index's shards is listed VERIFIED with zero mismatches and zero missing --
    never "every file listed so far is fine"."""
    files = got.get("files", [])
    verified = sum(1 for f in files if f.get("status") == "VERIFIED")
    mismatch = int(got.get("mismatch", sum(1 for f in files if f.get("status") == "MISMATCH")))
    missing = int(got.get("missing", sum(1 for f in files if f.get("status") == "MISSING")))
    complete = verified >= expected_files and expected_files > 0
    return {"path": path, "present": True, "files_listed": len(files), "files_verified": verified,
            "files_expected": int(expected_files), "mismatch": mismatch, "missing": missing,
            "partial": not complete,
            "all_verified": bool(complete and mismatch == 0 and missing == 0),
            "updated_utc": got.get("updated_utc"), "receipt_sha256": sha256_file(path)}


# --------------------------------------------------------------------------
# share policy (test 4)
# --------------------------------------------------------------------------
def read_share_rate(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        got = json.load(handle)
    for key in ("MB_per_s", "mb_per_s", "rate_MB_per_s", "megabytes_per_second"):
        if key in got:
            return {"path": path, "MB_per_s": float(got[key]), "receipt_sha256": sha256_file(path)}
    if isinstance(got.get("dd"), dict) and "MB_per_s" in got["dd"]:
        return {"path": path, "MB_per_s": float(got["dd"]["MB_per_s"]), "receipt_sha256": sha256_file(path)}
    raise SystemExit(f"share_rate receipt {path} carries no MB_per_s field: {list(got)[:12]}")


def resolve_share_policy(src_dir: str, *, no_share_read: bool | None, allow_share_read: bool,
                         share_rate_receipt: str, share_root: str = SHARE_ROOT) -> dict[str, Any]:
    """Refuse a share read without a measured rate; default to no-share-read off the share.

    * source NOT under the share: ``no_share_read`` resolves to True unless the
      caller said otherwise; no receipt is consulted.
    * source under the share and ``--no-share-read``: contradiction, refused.
    * source under the share: the share_rate receipt must exist and read
      >= 1 MB/s (or ``--allow-share-read`` records the override).
    """
    on_share = under_share(src_dir, share_root)
    policy: dict[str, Any] = {"share_root": share_root, "read_source": os.path.realpath(src_dir),
                              "source_on_share": on_share}
    if not on_share:
        policy["no_share_read"] = True if no_share_read is None else bool(no_share_read)
        policy["share_rate"] = None
        policy["note"] = "read source is a local disk; the share_rate receipt is not consulted"
        return policy
    if no_share_read:
        raise SystemExit(f"--no-share-read was asked for but the read source {src_dir} is under {share_root}; "
                         "point --src-dir at the Lexar copy or drop the flag")
    policy["no_share_read"] = False
    if allow_share_read:
        policy["share_rate"] = read_share_rate(share_rate_receipt) if os.path.exists(share_rate_receipt) else None
        policy["note"] = "--allow-share-read recorded; the rate check was overridden"
        return policy
    if not os.path.exists(share_rate_receipt):
        raise SystemExit(f"the read source {src_dir} is on the share and the share_rate receipt "
                         f"{share_rate_receipt} is missing; measure the share first (run-book S0) or "
                         "read from the Lexar copy")
    rate = read_share_rate(share_rate_receipt)
    if rate["MB_per_s"] < MIN_SHARE_MB_PER_S:
        raise SystemExit(f"share_rate receipt {share_rate_receipt} reads {rate['MB_per_s']} MB/s < "
                         f"{MIN_SHARE_MB_PER_S} MB/s; a 1.42 TiB pass at that rate is not a plan.  "
                         "Read from the Lexar copy or pass --allow-share-read deliberately")
    policy["share_rate"] = rate
    return policy


# --------------------------------------------------------------------------
# the corpus: evidence + review segments with in-band masks
# --------------------------------------------------------------------------
class Segment:
    """One ids file: ``input_ids`` [rows, T] plus the in-band split masks."""

    def __init__(self, path: str, role: str):
        from safetensors.torch import load_file

        got = load_file(path)
        self.path = path
        self.role = role
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.metadata = read_metadata(path)
        ids = got["input_ids"].to(torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2:
            raise SystemExit(f"{path}: input_ids must be [rows, T], got {list(ids.shape)}")
        self.ids = ids.contiguous()
        self.rows, self.T = int(ids.shape[0]), int(ids.shape[1])
        missing = [k for k in ("fit_mask", "held_mask", "real_lengths") if k not in got]
        if missing:
            raise SystemExit(f"{path} carries no in-band split ({missing} missing); this pass fits on the "
                             "corpus' own fit_mask/held_mask (k3_corpus_build v2 layout) and refuses to "
                             "invent a holdout")
        self.real_lengths = got["real_lengths"].to(torch.long)
        self.fit_mask = got["fit_mask"].to(torch.bool)
        self.held_mask = got["held_mask"].to(torch.bool)
        self.split = got["split"].to(torch.long) if "split" in got else None
        if self.fit_mask.shape != self.ids.shape or self.held_mask.shape != self.ids.shape:
            raise SystemExit(f"{path}: fit_mask/held_mask must be [rows, T]")
        if bool((self.fit_mask & self.held_mask).any()):
            raise SystemExit(f"{path}: fit_mask and held_mask overlap")
        positions = torch.arange(self.T).unsqueeze(0)
        self.real_mask = positions < self.real_lengths.unsqueeze(1)
        if role == "review":
            if bool(self.fit_mask.any()):
                raise SystemExit(f"{path}: a review file must carry an all-False fit_mask (masked rows)")
            self.held_mask = torch.zeros_like(self.held_mask)
        self.input_ids_sha256 = sha256_bytes(self.ids.contiguous().numpy().tobytes())
        self.fit_mask_sha256 = sha256_bytes(self.fit_mask.contiguous().numpy().tobytes())
        self.held_mask_sha256 = sha256_tensor(self.held_mask)
        self.real_lengths_sha256 = sha256_tensor(self.real_lengths)
        self.split_sha256 = sha256_tensor(self.split) if self.split is not None else None

    def record(self) -> dict[str, Any]:
        return {
            "path": self.path, "role": self.role, "rows": self.rows, "sequence_length": self.T,
            "tokens": self.rows * self.T, "real_tokens": int(self.real_mask.sum()),
            "fit_tokens": int(self.fit_mask.sum()), "held_tokens": int(self.held_mask.sum()),
            "input_ids_sha256": self.input_ids_sha256, "fit_mask_sha256": self.fit_mask_sha256,
            "held_mask_sha256": self.held_mask_sha256,
            "real_lengths_sha256": self.real_lengths_sha256, "split_sha256": self.split_sha256,
            "file_sha256": sha256_file(self.path),
            "metadata": {k: (v if len(v) < 200 else v[:200] + "...") for k, v in self.metadata.items()},
        }


class Corpus:
    """Every segment, flattened token-major: evidence first, review after."""

    def __init__(self, evidence: str, review: list[str]):
        self.segments = [Segment(evidence, "evidence")] + [Segment(p, "review") for p in review]
        self.offsets: list[int] = []
        total = 0
        for seg in self.segments:
            self.offsets.append(total)
            total += seg.rows * seg.T
        self.tokens = total
        self.fit = torch.cat([s.fit_mask.flatten() for s in self.segments])
        self.held = torch.cat([s.held_mask.flatten() for s in self.segments])
        self.real = torch.cat([s.real_mask.flatten() for s in self.segments])
        self.review = torch.cat([
            (s.real_mask if s.role == "review" else torch.zeros_like(s.real_mask)).flatten()
            for s in self.segments])
        # the provenance stamp every snapshot carries and every resume checks
        self.sha256 = sha256_text(canonical_json(
            [{"file": s.name, "role": s.role, "input_ids_sha256": s.input_ids_sha256,
              "shape": [s.rows, s.T], "fit_mask_sha256": s.fit_mask_sha256,
              "held_mask_sha256": s.held_mask_sha256, "real_lengths_sha256": s.real_lengths_sha256,
              "split_sha256": s.split_sha256} for s in self.segments]))

    def record(self) -> dict[str, Any]:
        return {"corpus_sha256": self.sha256, "tokens": self.tokens,
                "fit_tokens": int(self.fit.sum()), "held_tokens": int(self.held.sum()),
                "real_tokens": int(self.real.sum()), "review_tokens": int(self.review.sum()),
                "segments": [s.record() for s in self.segments]}

    def row_chunks(self, row_chunk: int):
        """(segment, row0, row1, flat_start, flat_stop) over every segment."""
        for seg, offset in zip(self.segments, self.offsets):
            for r0 in range(0, seg.rows, max(1, row_chunk)):
                r1 = min(r0 + max(1, row_chunk), seg.rows)
                yield seg, r0, r1, offset + r0 * seg.T, offset + r1 * seg.T


# --------------------------------------------------------------------------
# shard staging with a rolling window (test 3)
# --------------------------------------------------------------------------
class ShardStager:
    """Copy shards to a local window ahead of use; delete ONLY what this run staged.

    A manifest in the stage dir records every file this tool copied, with the
    run id that copied it.  ``release`` deletes a shard only when (a) this run
    created it, (b) ``--delete-staged-after-consume`` is on, and (c) both
    streams have consumed every layer that needs it and no upcoming layer in
    the plan does.  A file that merely sits in the stage dir under a shard's
    name -- placed by someone else, or by an earlier run id -- is never used
    (its bytes are not this run's to vouch for) and never deleted.
    """

    def __init__(self, src_dir: str, stage_dir: str | None, *, enabled: bool, window: int, prefetch: int,
                 delete_after_consume: bool, run_id: str, stall_seconds: float = DEFAULT_STAGE_STALL_SECONDS):
        self.src_dir = src_dir
        self.stage_dir = stage_dir
        self.enabled = bool(enabled and stage_dir)
        self.window = max(1, int(window))
        self.prefetch_n = max(0, int(prefetch))
        self.delete_after_consume = bool(delete_after_consume)
        self.stall_seconds = float(stall_seconds)
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.owned: dict[str, dict[str, Any]] = {}
        self.foreign: dict[str, dict[str, Any]] = {}
        self._pending: list[str] = []
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = False
        self._error: BaseException | None = None
        # copy-in-flight progress, read by the path_for watchdog
        self._copy_bytes = 0
        self._copy_last_progress = time.time()
        if self.enabled:
            os.makedirs(stage_dir, exist_ok=True)
            self._load_manifest()

    # -- manifest ----------------------------------------------------------
    def _manifest_path(self) -> str:
        return os.path.join(self.stage_dir, STAGING_MANIFEST)

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            prior = json.load(handle)
        for shard, rec in prior.get("staged", {}).items():
            if os.path.exists(os.path.join(self.stage_dir, shard)):
                self.foreign[shard] = rec

    def _write_manifest(self) -> None:
        payload = {"tool": TOOL_NAME, "staged": {**self.foreign, **self.owned}}
        write_receipt(self._manifest_path(), payload)

    # -- worker ------------------------------------------------------------
    def _resident_owned(self) -> int:
        """Shards this run staged that still hold a WINDOW slot: ready and not yet consumed.

        A consumed shard kept on disk (``--delete-staged-after-consume`` off, state
        ``consumed_kept``) no longer counts: it is nobody's blocker.  Model-level
        shards never enter here (``DualStreamRun._model_tensor`` reads the source).
        """
        return sum(1 for rec in self.owned.values() if rec.get("state") == "ready")

    def _copy_in_flight(self) -> str | None:
        return next((s for s, rec in self.owned.items() if rec.get("state") == "copying"), None)

    def _worker(self) -> None:
        try:
            while True:
                with self._cv:
                    while not self._pending and not self._stop:
                        self._cv.wait()
                    if self._stop and not self._pending:
                        return
                    shard = self._pending[0]
                    while self._resident_owned() >= self.window and not self._stop:
                        self._cv.wait()
                    if self._stop:
                        return
                    self._pending.pop(0)
                    self.owned[shard] = {"run_id": self.run_id, "state": "copying"}
                    self._copy_bytes = 0
                    self._copy_last_progress = time.time()
                src = os.path.join(self.src_dir, shard)
                dst = os.path.join(self.stage_dir, shard)
                tmp = f"{dst}.part{os.getpid()}"
                started = time.time()
                with open(src, "rb", buffering=COPY_CHUNK) as fin, open(tmp, "wb", buffering=COPY_CHUNK) as fout:
                    while True:
                        block = fin.read(COPY_CHUNK)
                        if not block:
                            break
                        fout.write(block)
                        with self._cv:  # progress the watchdog can see
                            self._copy_bytes += len(block)
                            self._copy_last_progress = time.time()
                os.replace(tmp, dst)
                size = os.path.getsize(dst)
                src_size = os.path.getsize(src)
                elapsed = max(time.time() - started, 1e-9)
                with self._cv:
                    if size != src_size:
                        self.owned[shard]["state"] = "size_mismatch"
                        self._error = SystemExit(f"staged copy of {shard} is {size} bytes, source {src_size}")
                        self._cv.notify_all()
                        return
                    self.owned[shard].update({"state": "ready", "bytes": size, "seconds": round(elapsed, 1),
                                              "MB_per_s": round(size / 1e6 / elapsed, 1),
                                              "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    self.events.append({"shard": shard, "action": "staged", "bytes": size,
                                        "seconds": round(elapsed, 1), "MB_per_s": round(size / 1e6 / elapsed, 1),
                                        "run_id": self.run_id})
                    self._write_manifest()
                    self._cv.notify_all()
        except BaseException as exc:  # surfaced to the main thread by path_for
            with self._cv:
                self._error = exc
                self._cv.notify_all()

    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="k3-stager", daemon=True)
            self._thread.start()

    def queue(self, shards: list[str]) -> None:
        """Ask for these shards, in order, without waiting."""
        if not self.enabled:
            return
        with self._cv:
            for shard in shards:
                if shard in self.owned or shard in self._pending or shard in self.foreign:
                    continue
                if os.path.exists(os.path.join(self.stage_dir, shard)):
                    # a file under the shard's name that this run did not create and the
                    # manifest does not know: never overwritten, never used, never deleted
                    self.foreign[shard] = {"run_id": None, "state": "unmanaged_file_present"}
                    self.events.append({"shard": shard, "action": "unmanaged_file_left_alone",
                                        "run_id": self.run_id})
                    continue
                self._pending.append(shard)
            self._cv.notify_all()
        self._ensure_thread()

    def path_for(self, shard: str) -> str:
        """The staged path when this run staged it (waiting for the copy); else the source."""
        src = os.path.join(self.src_dir, shard)
        if not self.enabled:
            return src
        if shard in self.foreign:
            # not ours: neither used nor deleted
            self.events.append({"shard": shard, "action": "foreign_file_ignored", "run_id": self.run_id,
                                "note": "a staged file this run did not create is read from the source instead"})
            return src
        self.queue([shard])
        with self._cv:
            while True:
                if self._error is not None:
                    raise self._error
                rec = self.owned.get(shard)
                if rec and rec.get("state") in ("ready", "consumed_kept"):
                    return os.path.join(self.stage_dir, shard)
                self._check_stalled(shard)
                self._cv.wait(timeout=5.0)

    def _check_stalled(self, shard: str) -> None:
        """The watchdog (called under ``_cv``): a wait that cannot end is a named refusal, not a hang.

        * STRUCTURAL: ``shard`` is still queued, the window is full and no copy is in
          flight.  Only the main thread releases window slots and the main thread is the
          one waiting here, so nothing will ever move -- refuse now, naming the residents.
        * PROGRESS: a copy is in flight but has not advanced by a byte for
          ``stall_seconds`` (the share stopped answering) -- refuse, naming the shard.
        """
        copying = self._copy_in_flight()
        if shard in self._pending and copying is None and self._resident_owned() >= self.window:
            resident = sorted(s for s, r in self.owned.items() if r.get("state") == "ready")
            raise SystemExit(
                f"stager deadlock: {shard} is queued behind a full window of {self.window} "
                f"(resident, unreleased: {resident}) with no copy in flight; a release cannot happen while "
                "the pass waits for this shard.  Raise --stage-window above the number of shards one layer "
                "straddles, or check that every consumed shard is released")
        if copying is not None and time.time() - self._copy_last_progress > self.stall_seconds:
            raise SystemExit(
                f"stager stalled: the copy of {copying} has not advanced past {self._copy_bytes} bytes for "
                f"{self.stall_seconds:.0f}s (--stage-stall-seconds); the source stopped answering")

    def note_source_read(self, shard: str, name: str) -> None:
        """A model-level tensor read straight from the source: recorded, never staged."""
        self.events.append({"shard": shard, "tensor": name, "action": "model_level_read_from_source",
                            "run_id": self.run_id})

    def release(self, consumed: list[str], still_needed: set[str]) -> list[dict[str, Any]]:
        """After BOTH streams finished a layer: drop what this run staged and nothing else needs.

        With ``--delete-staged-after-consume`` off a consumed shard stays on disk but
        leaves the window (state ``consumed_kept``), so the worker is never blocked
        by files that have already done their job.
        """
        released: list[dict[str, Any]] = []
        if not self.enabled:
            return released
        with self._cv:
            for shard in consumed:
                rec = self.owned.get(shard)
                if rec is None or rec.get("state") != "ready":
                    continue
                if shard in still_needed:
                    event = {"shard": shard, "action": "kept_still_needed", "run_id": self.run_id}
                elif not self.delete_after_consume:
                    rec["state"] = "consumed_kept"
                    self._write_manifest()
                    event = {"shard": shard, "action": "kept_no_delete_flag", "run_id": self.run_id,
                             "note": "left on disk, no longer counted against the window"}
                else:
                    path = os.path.join(self.stage_dir, shard)
                    size = os.path.getsize(path) if os.path.exists(path) else None
                    os.remove(path)
                    del self.owned[shard]
                    event = {"shard": shard, "action": "deleted_after_both_streams_consumed",
                             "bytes": size, "run_id": self.run_id}
                    self._write_manifest()
                self.events.append(event)
                released.append(event)
            self._cv.notify_all()
        return released

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def record(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "stage_dir": self.stage_dir, "window": self.window,
                "prefetch": self.prefetch_n, "delete_after_consume": self.delete_after_consume,
                "stall_seconds": self.stall_seconds,
                "run_id": self.run_id, "events": list(self.events),
                "still_staged_by_this_run": sorted(s for s, r in self.owned.items()
                                                   if r.get("state") in ("ready", "consumed_kept")),
                "window_residents": sorted(s for s, r in self.owned.items() if r.get("state") == "ready"),
                "model_level_shards": "read from the source, never staged (see model_level_read_from_source events)",
                "foreign_files_left_alone": sorted(self.foreign)}


# --------------------------------------------------------------------------
# layer reads through the stager, with per-shard MB/s
# --------------------------------------------------------------------------
def read_layer(index: K3Index, layer_idx: int, stager: ShardStager, device: torch.device,
               experts_device: torch.device, compute_dtype: torch.dtype | None,
               progress_tag: str | None = None, keep_attention: bool = False,
               ) -> tuple[LayerWeights, dict[str, torch.Tensor], dict[str, Any]]:
    """`k3_layer.load_layer` routed through the stager; returns (weights, raw body copy, timing).

    The raw copy carries the on-disk (uncast, host) non-expert tensors the form-E
    file and the layer-0 file take verbatim; with ``keep_attention`` the attention
    tensors stay in it too (``--body-dir`` writes them verbatim).
    """
    keys = index.layer_keys(layer_idx)
    if not keys:
        raise SystemExit(f"layer {layer_idx}: the index carries no tensors under {index.layer_prefix_for(layer_idx)}")
    tensors: dict[str, torch.Tensor] = {}
    per_shard = []
    total_bytes = 0
    started = time.time()
    for shard in index.shards_for(keys):
        wanted = [k for k in keys if index.weight_map[k] == shard]
        path = stager.path_for(shard)
        shard_start = time.time()
        got = read_tensors(path, wanted)
        got_bytes = sum(t.numel() * t.element_size() for t in got.values())
        elapsed = max(time.time() - shard_start, 1e-9)
        tensors.update(got)
        total_bytes += got_bytes
        per_shard.append({"shard": shard, "path": path, "tensors": len(got),
                          "gigabytes": round(got_bytes / 1e9, 3), "seconds": round(elapsed, 1),
                          "MB_per_s": round(got_bytes / 1e6 / elapsed, 1),
                          "staged": path != os.path.join(stager.src_dir, shard)})
        if progress_tag:
            log(f"{progress_tag}: {shard} {len(got)} tensors {got_bytes/1e9:.2f} GB in {elapsed:.0f}s = "
                f"{got_bytes/1e6/elapsed:.1f} MB/s")
    missing = [k for k in keys if k not in tensors]
    if missing:
        raise SystemExit(f"index promised {len(missing)} tensors the shards do not carry, e.g. {missing[:3]}")
    wall = max(time.time() - started, 1e-9)
    timing = {"tensors": len(tensors), "bytes": total_bytes, "gigabytes": round(total_bytes / 1e9, 3),
              "seconds": round(wall, 1), "MB_per_s": round(total_bytes / 1e6 / wall, 1), "shards": per_shard}
    # the on-disk (uncast) copies of the tensors the form-E file carries verbatim
    prefix = index.layer_prefix_for(layer_idx)
    raw_body = {k[len(prefix):]: v for k, v in tensors.items()
                if k.startswith(prefix) and ".experts." not in k
                and (keep_attention or not k[len(prefix):].startswith("self_attn."))}
    weights = LayerWeights(layer_idx, tensors, index)
    weights.to(device, experts_device)
    resolved, dtype_source = resolve_compute_dtype(weights, compute_dtype)
    for group in (weights.attn, weights.res, weights.body):
        for key, tensor in group.items():
            if tensor.is_floating_point() and tensor.dtype != resolved:
                group[key] = tensor.to(resolved)
    timing["compute_dtype"] = str(resolved)
    timing["compute_dtype_source"] = dtype_source
    return weights, raw_body, timing


def resolve_compute_dtype(weights: LayerWeights, compute_dtype: torch.dtype | None) -> tuple[torch.dtype, str]:
    """``--compute-dtype auto`` = the PARENT's dtype: what its projection matrices carry.

    Kimi-K3 ships a MIXED checkpoint: the projections are BF16 but
    ``self_attn.{q,k,v}_conv1d.weight``, ``A_log``, ``dt_bias``, ``o_norm.weight``
    and ``gate.e_score_correction_bias`` are F32 (shard headers, layers 0-1).
    ``transformers`` loading at ``torch_dtype=bfloat16`` casts every floating
    parameter to bf16 (``modeling_kimi_k3.py`` keeps nothing in fp32), so THAT
    is the parent the teacher pair reproduces.  Left uncast, ``F.conv1d`` refuses
    the bf16 activation against the F32 kernel ("expected scalar type BFloat16
    but found Float") -- the 2026-09-06 L0-1 launch on GPU6 died there; the
    fixtures never saw it because ``k3_synth`` writes bf16 conv kernels.  The
    fp32 gate / recurrence math in ``k3_layer`` is unchanged: it ``.float()``s
    its inputs either way.  The uncast on-disk copies (``raw_body``) still feed
    every verbatim write and sha match.
    """
    if compute_dtype is not None:
        return compute_dtype, "--compute-dtype"
    for group_name, group in (("attn", weights.attn), ("body", weights.body)):
        for key, tensor in group.items():
            if key.endswith("proj.weight") and tensor.is_floating_point():
                return tensor.dtype, f"parent projection dtype ({group_name}.{key})"
    raise SystemExit(f"layer {weights.layer_idx}: no floating projection weight to take the parent dtype from")


# --------------------------------------------------------------------------
# the layer, in halves, chunked over rows (attention) and tokens (everything else)
# --------------------------------------------------------------------------
def _chunked_linear(x: torch.Tensor, weight: torch.Tensor, chunk: int) -> torch.Tensor:
    out = x.new_empty(x.shape[0], weight.shape[0])
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        out[start:stop] = F.linear(x[start:stop], weight)
    return out


class Pair:
    """One residual pair: ``prefix`` [N, H] and ``blocks`` [N, b, H], on the residual device."""

    __slots__ = ("prefix", "blocks")

    def __init__(self, prefix: torch.Tensor, blocks: torch.Tensor):
        self.prefix = prefix
        self.blocks = blocks

    def clone(self) -> "Pair":
        return Pair(self.prefix.clone(), self.blocks.clone())


def teacher_temporary_lifetime(prefix: torch.Tensor, persisted_prefix: torch.Tensor,
                               taps: dict[str, Any]) -> dict[str, Any]:
    """Drop the unused teacher latent tap; caller must also delete its prefix alias.

    Fit consumes the FOLDED latent and the teacher y_lat/router, never this tap.
    Report logical reference sizes, not an unmeasured allocator-memory saving.
    A same-device residual Pair may still own the prefix storage legitimately.
    """
    latent = taps.pop("latent_in", None)
    released = {name: {"logical_bytes": value.numel() * value.element_size(),
                       "device": str(value.device), "dtype": str(value.dtype)}
                for name, value in (("teacher_prefix_local_alias", prefix), ("unused_teacher_latent_in", latent))
                if value is not None}
    return {"released_references": released,
            "prefix_storage_retained_by_residual_pair":
                prefix.device == persisted_prefix.device and
                prefix.untyped_storage().data_ptr() == persisted_prefix.untyped_storage().data_ptr(),
            "cuda_allocator_bytes_freed": None,
            "note": "reference logical sizes only; CPU copies preserve dtype/values; no allocator measurement"}


def attention_half(runner: StreamedLayer, pair: Pair, corpus: Corpus, device: torch.device, *,
                   row_chunk: int, token_chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`_forward_attn_residual` up to the FFN input.  Returns (carried, ffn_in, blocks_after).

    ``carried`` and ``ffn_in`` are [N, H] on ``device``; ``blocks_after`` is the
    residual stack on the pair's own device (with this layer's snapshot pushed
    when ``layer_idx % block_size == 0``).  Token-wise steps run through the
    token-chunked ``apply_attn_res``; the attention runs per row chunk so the
    [heads, T, T] score matrix is bounded by ``--row-chunk`` rows.
    """
    w = runner.w
    eps = runner.eps
    hidden = int(pair.prefix.shape[1])
    snapshot = (w.layer_idx % runner.block_size) == 0
    carried_all = torch.empty(corpus.tokens, hidden, dtype=pair.prefix.dtype, device=device)
    ffn_in_all = torch.empty_like(carried_all)
    for seg, r0, r1, start, stop in corpus.row_chunks(row_chunk):
        n_rows = r1 - r0
        p = pair.prefix[start:stop].to(device, non_blocking=False)
        bl = pair.blocks[start:stop].to(device)
        if bl.shape[1] > 0:
            hidden_states = apply_attn_res(p, bl, w.res["self_attention_res_proj.weight"],
                                           w.res["self_attention_res_norm.weight"], eps,
                                           token_chunk_size=token_chunk)
        else:
            hidden_states = p
        if snapshot:
            bl = torch.cat([bl, p.unsqueeze(1)], dim=1)
        normed = rms_norm(hidden_states, w.res["input_layernorm.weight"], eps).view(n_rows, seg.T, hidden)
        attn_out, _ = runner.attention(normed, state=None)
        attn_out = attn_out.reshape(-1, hidden)
        carried = attn_out if snapshot else p + attn_out
        h2 = apply_attn_res(carried, bl, w.res["mlp_res_proj.weight"], w.res["mlp_res_norm.weight"], eps,
                            token_chunk_size=token_chunk)
        h2 = rms_norm(h2, w.res["post_attention_layernorm.weight"], eps)
        carried_all[start:stop] = carried
        ffn_in_all[start:stop] = h2
        del p, bl, hidden_states, normed, attn_out, carried, h2
    blocks_after = pair.blocks
    if snapshot:
        blocks_after = torch.cat([pair.blocks, pair.prefix.unsqueeze(1)], dim=1)
    return carried_all, ffn_in_all, blocks_after


def routed_y_lat(weights: LayerWeights, x: torch.Tensor, latent: torch.Tensor, cfg: dict,
                 device: torch.device) -> tuple[torch.Tensor, dict[str, Any], float]:
    """The teacher's routed accumulator over ALL tokens, each expert dequantized ONCE.

    The arithmetic is `k3_layer.moe_forward`'s routed loop verbatim; the one
    addition is the ``.to(device)`` after the dequant so the PACKED experts can
    stay on ``--experts-device cpu`` (what the run book asks for) while the
    tokens are on the card.  Returns (y_lat fp32 [N, latent], router taps,
    dequant seconds).
    """
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    topk_idx, topk_weight, _ = route(x, weights, cfg)
    y_lat = torch.zeros(latent.shape[0], latent.shape[1], dtype=torch.float32, device=latent.device)
    counts = torch.bincount(topk_idx.flatten(), minlength=int(cfg["num_experts"]))
    touched = torch.nonzero(counts).flatten().tolist()
    dequant_seconds = 0.0
    for expert in touched:
        hit = (topk_idx == expert)
        rows = hit.any(dim=1).nonzero().flatten()
        weight = topk_weight[rows].masked_fill(~hit[rows], 0.0).sum(dim=1)
        started = time.time()
        ex = dequant_expert(weights, expert, dtype=latent.dtype)
        ex = {k: v.to(device) for k, v in ex.items()}
        dequant_seconds += time.time() - started
        chunk = latent[rows]
        activated = situ(F.linear(chunk, ex["w1"]), F.linear(chunk, ex["w3"]), beta, linear_beta)
        y_lat[rows] += F.linear(activated, ex["w2"]).float() * weight.unsqueeze(-1)
        del ex, chunk, activated
    taps = {
        "router_topk_idx": topk_idx.detach().to(torch.int16),
        "router_topk_w": topk_weight.detach().float(),
        "router_counts": counts.detach().cpu(),
        "experts_touched": int(len(touched)),
    }
    return y_lat, taps, dequant_seconds


def dense_y_lat(dense: dict[str, torch.Tensor], latent: torch.Tensor, cfg: dict, chunk: int) -> torch.Tensor:
    """`moe_forward`'s dense branch (the form-E swap-in), token-chunked."""
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    w1 = dense["w1_s"].to(latent.device, latent.dtype)
    w3 = dense["w3_s"].to(latent.device, latent.dtype)
    w2p = dense["w2p"].to(latent.device)
    out = torch.empty(latent.shape[0], w2p.shape[0], dtype=torch.float32, device=latent.device)
    for start in range(0, latent.shape[0], chunk):
        stop = min(start + chunk, latent.shape[0])
        piece = latent[start:stop]
        activated = situ(F.linear(piece, w1), F.linear(piece, w3), beta, linear_beta)
        out[start:stop] = F.linear(activated, w2p.to(activated.dtype)).float()
    return out


def moe_write(weights: LayerWeights, x: torch.Tensor, y_lat: torch.Tensor, cfg: dict, chunk: int) -> torch.Tensor:
    """`routed_expert_norm` -> `routed_expert_up_proj` + the two shared experts, token-chunked."""
    eps = float(cfg["rms_norm_eps"])
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    body = weights.body
    out = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        flat = x[start:stop]
        y = rms_norm(y_lat[start:stop].to(flat.dtype), body["block_sparse_moe.routed_expert_norm.weight"], eps)
        routed = F.linear(y, body["block_sparse_moe.routed_expert_up_proj.weight"])
        shared = F.linear(
            situ(F.linear(flat, body["block_sparse_moe.shared_experts.gate_proj.weight"]),
                 F.linear(flat, body["block_sparse_moe.shared_experts.up_proj.weight"]), beta, linear_beta),
            body["block_sparse_moe.shared_experts.down_proj.weight"],
        )
        out[start:stop] = routed + shared
    return out


def ffn_half(runner: StreamedLayer, carried: torch.Tensor, ffn_in: torch.Tensor, cfg: dict, *,
             device: torch.device, token_chunk: int, dense: dict[str, torch.Tensor] | None = None,
             latent: torch.Tensor | None = None, y_lat: torch.Tensor | None = None,
             ) -> tuple[torch.Tensor, dict[str, Any]]:
    """The FFN half of `_forward_attn_residual`; returns (new prefix [N, H] on device, taps).

    MoE layer: ``latent`` may be handed in (the folded pair's `latent_in`, which
    the fit already needed) and ``y_lat`` may be handed in (a precomputed
    accumulator); otherwise they are computed here -- routed (teacher) or dense
    (``dense`` given).
    """
    w = runner.w
    taps: dict[str, Any] = {}
    if not w.is_moe:
        out = torch.empty_like(ffn_in)
        for start in range(0, ffn_in.shape[0], token_chunk):
            stop = min(start + token_chunk, ffn_in.shape[0])
            out[start:stop] = dense_mlp_forward(w, ffn_in[start:stop], cfg)
        return carried + out, taps
    if latent is None:
        latent = _chunked_linear(ffn_in, w.body["block_sparse_moe.routed_expert_down_proj.weight"], token_chunk)
    taps["latent_in"] = latent
    if y_lat is None:
        if dense is not None:
            y_lat = dense_y_lat(dense, latent, cfg, token_chunk)
            taps["dense_channels"] = int(dense["w1_s"].shape[0])
        else:
            y_lat, router, dequant_seconds = routed_y_lat(w, ffn_in, latent, cfg, device)
            taps.update(router)
            taps["dequant_seconds"] = dequant_seconds
    taps["y_lat"] = y_lat
    out = moe_write(w, ffn_in, y_lat, cfg, token_chunk)
    return carried + out, taps


# --------------------------------------------------------------------------
# the measured acceptance rule (v1.2.0)
# --------------------------------------------------------------------------
class Refusal(SystemExit):
    """A named refusal that carries its numbers (``detail``) into the partial receipt."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.detail = dict(detail or {})


def measured_solve_gram(
    gram_inner: torch.Tensor, cross_inner: torch.Tensor,
    gram_check: torch.Tensor, cross_check: torch.Tensor, check_y_sq: float,
    prior: torch.Tensor, *, fit_y_sq: float | None = None, stash: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """`c14.law_abiding_solve_gram` with the admission rule removed and the ladder measured.

    Same three passes, same `LAMBDA_LADDER`, same `prior_anchored_gram` arithmetic
    (lam = rel x mean diag G, rhs = B + lam prior^T, Cholesky), same in-place
    consumption of the inputs.  What changes:

      * lambda = argmin check-split SSE over ALL nine rungs; ``update_scale`` no
        longer admits or vetoes.  THE LAW's verdict is recorded verbatim under
        ``law_verdict`` (``would_refuse``, its exact message) so the old ruling is
        never lost, only demoted from a veto to a measurement.
      * every rung records ``check_relL2`` (sqrt(check_sse / ||Y_check||^2)),
        ``solution_norm``, ``update_scale`` and -- when ``fit_y_sq`` is given --
        the full-fit ``fit_relL2`` from the Gram identity, so the rel=1e-6 free
        solve's fit error (the memoriser reference) is on the receipt.
      * ``stash["prior"]`` receives the prior so the caller can SCORE it on the
        held rows (the baseline the installed map must beat).

    Why: on a routed-expert union the prior is the parent column x the channel's
    mean router mass (~1e-3 on real K3), so ||W2' - prior|| / ||prior|| is a norm
    ratio against a near-zero reference -- 81.642 at L01 for a solution whose
    held-out relL2 the veto never let anyone see.
    """
    ladder_rels = tuple(c14.LAMBDA_LADDER)
    reference = float(prior.norm())
    if stash is not None:
        stash["prior"] = prior.detach().clone()

    # -- 1. the check split --------------------------------------------------
    check_sse: list[float] = []
    for rel in ladder_rels:
        inner = c14.prior_anchored_gram(gram_inner, cross_inner, prior, rel=rel)
        check_sse.append(c14._check_sse(inner, gram_check, cross_check, check_y_sq))
        del inner

    # -- fold check into inner, in place: now they ARE the full-fit system ----
    gram_inner.add_(gram_check)
    cross_inner.add_(cross_check)
    gram_all, cross_all = gram_inner, cross_inner

    # -- 2. the full-fit ladder, MEASURED (no veto) ------------------------------
    rows: list[dict[str, Any]] = []
    for rel, sse in zip(ladder_rels, check_sse):
        full = c14.prior_anchored_gram(gram_all, cross_all, prior, rel=rel)
        diff = float((full - prior).norm())
        row: dict[str, Any] = {
            "lambda_relative": rel,
            "check_sse": sse,
            "check_relL2": (max(sse, 0.0) / check_y_sq) ** 0.5 if check_y_sq > 0 else None,
            "update_scale": (diff / reference) if reference > 0 else None,
            "solution_norm": float(full.norm()),
            "solution_minus_prior_norm": diff,
        }
        if fit_y_sq is not None:
            fit_sse = c14._check_sse(full, gram_all, cross_all, fit_y_sq)
            row["fit_sse"] = fit_sse
            row["fit_relL2"] = (max(fit_sse, 0.0) / fit_y_sq) ** 0.5 if fit_y_sq > 0 else None
        rows.append(row)
        del full

    chosen_i = min(range(len(rows)), key=lambda i: rows[i]["check_sse"])
    chosen = rows[chosen_i]
    admitted = [row for row in rows if row["update_scale"] is not None and row["update_scale"] <= 1.0]
    free_scale = rows[0]["update_scale"]
    law_pick = min(admitted, key=lambda row: row["check_sse"]) if admitted else None
    would_refuse = not admitted
    law_message = None
    if would_refuse:
        free_txt = f"{free_scale:.3f}" if free_scale is not None else "inf (zero prior)"
        law_message = (f"no lambda on the ladder brings update_scale to 1.0 (free {free_txt}); "
                       "THE LAW refuses the install")
    monotone = all(check_sse[i] >= check_sse[i + 1] - 1e-12 for i in range(len(check_sse) - 1))

    # -- 3. the one solution that leaves this function -----------------------------
    solved = c14.prior_anchored_gram(gram_all, cross_all, prior, rel=chosen["lambda_relative"])
    record = {
        "update_scale": chosen["update_scale"],
        "lambda_relative": chosen["lambda_relative"],
        "free_solve_update_scale": free_scale,
        "lambda_selected_on": "internal fit split (argmin check_sse over the full ladder; no norm veto)",
        "lambda_law_would_have_picked": law_pick["lambda_relative"] if law_pick else None,
        "veto_binding": bool(free_scale > 1.0) if free_scale is not None else True,
        "lambdas_vetoed": int(len(rows) - len(admitted)),
        "chosen_is_largest_lambda": bool(chosen["lambda_relative"] == ladder_rels[-1]),
        "chosen_is_weakest_admitted": bool(law_pick is not None
                                           and chosen["lambda_relative"] == law_pick["lambda_relative"]),
        "check_sse_monotone_decreasing_in_lambda": bool(monotone),
        "interpretation": (
            "the check split carried no signal (monotone check_sse, largest lambda chosen): the installed "
            "map is the prior, shrunk -- read that as a NO-SIGNAL fold, not a healthy one"
            if (chosen["lambda_relative"] == ladder_rels[-1] and monotone)
            else "the check split discriminated between lambdas"
        ),
        "chooser": (f"measured ({TOOL_NAME} v{TOOL_VERSION}): argmin check_sse over all {len(rows)} rungs; "
                    "THE LAW's update_scale <= 1 admission is recorded under law_verdict and neither "
                    "chooses nor vetoes"),
        "law_verdict": {
            "rule": "c14.law_abiding_solve_gram: admit a rung only if ||W - prior||_F / ||prior||_F <= 1.0",
            "would_refuse": bool(would_refuse),
            "message": law_message,
            "free_solve_update_scale": free_scale,
            "min_update_scale": min((r["update_scale"] for r in rows if r["update_scale"] is not None),
                                    default=None),
            "reference_prior_norm": reference,
            "reference_prior_norm_zero": bool(reference == 0.0),
            "lambdas_admitted": [r["lambda_relative"] for r in admitted],
            "status": "RECORDED, not enforced (v1.2.0: the measured acceptance rule gates instead)",
        },
        "chosen_at_ladder_edge": bool(chosen_i in (0, len(rows) - 1)),
        "chosen_ladder_index": int(chosen_i),
        "prior_norm": reference,
        "check_y_sq": float(check_y_sq),
        "fit_y_sq": fit_y_sq,
        "free_fit_relL2": rows[0].get("fit_relL2"),
        "free_check_relL2": rows[0]["check_relL2"],
        "chosen_fit_relL2_gram": chosen.get("fit_relL2"),
        "chosen_check_relL2": chosen["check_relL2"],
        "ladder": rows,
    }
    return solved, record


def _score_relL2(block: dict[str, Any] | None, key: str = "relL2") -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    return float(value) if value is not None else None


def evaluate_acceptance(heldout: dict[str, Any], stream_held: dict[str, Any] | None,
                        zero_child_held: dict[str, Any] | None, *, mode: str = "gate",
                        constants: dict[str, Any] | None = None) -> dict[str, Any]:
    """The per-layer gates G1-G3 on the held rows (the lambda chooser never saw them).

    ``heldout`` is the rung receipt's ``heldout`` block (``DualCoresetFold.score``):
    ``y_lat_heldout`` / ``routed_write_heldout`` (fp32 W2'), the ``*_runtime_dtype``
    twins (W2' cast as the dense path casts it), ``y_lat_fit`` (the installed map on
    the fit rows) and ``baselines.{zero,prior}``.  ``stream_held`` is
    ``stream_divergence.held`` (folded-vs-teacher prefix), ``zero_child_held`` the
    same score for a child that writes y_lat = 0, on the same rows.

    A gate that cannot be evaluated (a missing number) FAILS by name; nothing here
    passes by default.
    """
    c = dict(ACCEPTANCE_CONSTANTS if constants is None else constants)
    floor = float(c["exact_fold_floor_relL2"])
    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, value, threshold, ok, note: str, **extra) -> None:
        gates[name] = {"value": value, "threshold": threshold, "pass": bool(ok), "note": note, **extra}

    y = _score_relL2(heldout.get("y_lat_heldout"))
    y_rt = _score_relL2(heldout.get("y_lat_heldout_runtime_dtype"))
    rw = _score_relL2(heldout.get("routed_write_heldout"))
    rw_rt = _score_relL2(heldout.get("routed_write_heldout_runtime_dtype"))
    baselines = heldout.get("baselines") or {}
    zero_y = _score_relL2((baselines.get("zero") or {}).get("y_lat_heldout"))
    zero_rw = _score_relL2((baselines.get("zero") or {}).get("routed_write_heldout"))
    prior_y = _score_relL2((baselines.get("prior") or {}).get("y_lat_heldout"))
    prior_rw = _score_relL2((baselines.get("prior") or {}).get("routed_write_heldout"))
    margin = float(c["G1_signal_margin"])

    # -- G1: signal on the held rows, fp32 and runtime dtype, vs the zero and prior maps --
    for name, val, val_rt, refs in (("G1_y_lat_heldout", y, y_rt, (zero_y, prior_y)),
                                    ("G1_routed_write_heldout", rw, rw_rt, (zero_rw, prior_rw))):
        available = [r for r in refs if r is not None]
        if val is None or not available:
            gate(name, val, None, False, "not evaluable: the held-out score or both baselines are missing")
            continue
        worst = max(v for v in (val, val_rt) if v is not None)
        threshold = margin * min(available)
        exact = worst <= floor
        gate(name, worst, threshold, exact or worst <= threshold,
             ("exact fold (held relL2 <= exact_fold_floor_relL2): on a uniform router the prior IS the answer, "
              "so a margin over it is not a bar" if exact
              else f"max(fp32, runtime dtype) held relL2 <= {margin} x min(zero, prior) baseline"),
             fp32=val, runtime_dtype=val_rt, baseline_zero=refs[0], baseline_prior=refs[1], exact_fold=exact)

    # -- G2: no memorisation -- held vs fit relL2 of the installed map ---------------------
    fit = _score_relL2(heldout.get("y_lat_fit"))
    ratio_max = float(c["G2_memorisation_ratio_max"])
    if y is None or fit is None:
        gate("G2_held_over_fit", None, ratio_max, False, "not evaluable: y_lat_fit was not scored")
    else:
        ratio = y / max(fit, 1e-12)
        exact = y <= floor
        gate("G2_held_over_fit", ratio, ratio_max, exact or ratio <= ratio_max,
             ("exact fold (held relL2 <= exact_fold_floor_relL2): ratio is noise over noise" if exact
              else "held y_lat relL2 / fit y_lat relL2 of the installed map"),
             held=y, fit=fit, exact_fold=exact)

    # -- G3: downstream -- the folded prefix must beat a child that writes nothing -----------
    s = _score_relL2(stream_held)
    z = _score_relL2(zero_child_held)
    prefix_margin = float(c["G3_prefix_margin"])
    if s is None or z is None:
        gate("G3_prefix_vs_zero_child", s, None, False,
             "not evaluable: stream_divergence.held or zero_child_prefix.held is missing")
    else:
        threshold = prefix_margin * z
        exact = s <= floor
        gate("G3_prefix_vs_zero_child", s, threshold, exact or s <= threshold,
             ("exact fold (prefix relL2 <= exact_fold_floor_relL2)" if exact
              else f"folded-vs-teacher prefix relL2 (held) <= {prefix_margin} x zero-child prefix relL2"),
             zero_child=z, exact_fold=exact)

    failing = [name for name, g in gates.items() if not g["pass"]]
    return {
        "version": ACCEPTANCE_VERSION,
        "mode": mode,
        "enforced": bool(mode == "gate"),
        "verdict": "accepted" if not failing else "refused",
        "failing": failing,
        "gates": gates,
        "constants": c,
        "calibrated_on": ACCEPTANCE_CALIBRATED_ON,
        "held_rows_used_by": "the gates only; lambda is chosen on the internal check split",
    }


def acceptance_message(layer_idx: int, rung: int, acceptance: dict[str, Any]) -> str:
    parts = []
    for name in acceptance["failing"]:
        g = acceptance["gates"][name]
        value = "missing" if g["value"] is None else f"{g['value']:.4f}"
        threshold = "n/a" if g["threshold"] is None else f"{g['threshold']:.4f}"
        parts.append(f"{name} {value} vs threshold {threshold} ({g['note']})")
    return (f"layer {layer_idx} K={rung} refused by the measured acceptance rule "
            f"({acceptance['version']}): " + "; ".join(parts) + "; nothing installed")


# --------------------------------------------------------------------------
# the fold on the corpus' own split (fit rows / held rows)
# --------------------------------------------------------------------------
class DualCoresetFold(c14.CoresetFold):
    """`CoresetFold` whose fit/held split is the corpus' in-band masks, with stage timings."""

    def __init__(self, weights: LayerWeights, cache: dict[str, torch.Tensor], cfg: dict, device: torch.device,
                 fit_mask: torch.Tensor, held_mask: torch.Tensor, check_blocks: int = 24, **kw):
        self._fit_mask = fit_mask.to(torch.bool).cpu()
        self._held_mask = held_mask.to(torch.bool).cpu()
        self.timings: dict[str, float] = {}
        super().__init__(weights, cache, cfg, device, holdout="corpus", holdout_blocks=check_blocks, **kw)

    def _split_positions(self, mode: str, blocks: int) -> None:
        if mode != "corpus":
            return super()._split_positions(mode, blocks)
        if self._fit_mask.numel() != self.tokens or self._held_mask.numel() != self.tokens:
            raise SystemExit(f"fit/held masks cover {self._fit_mask.numel()} tokens, the cache {self.tokens}")
        self.held = self._held_mask.clone()
        self.fit = self._fit_mask.clone()
        if bool((self.held & self.fit).any()):
            raise SystemExit("fit and held masks overlap")
        fit_positions = torch.nonzero(self.fit).flatten()
        n_fit = int(fit_positions.numel())
        size = max(1, -(-n_fit // max(int(blocks), 1)))
        chunk_id = torch.arange(n_fit) // size
        check_positions = fit_positions[(chunk_id % 3) == 2]
        self.check = torch.zeros(self.tokens, dtype=torch.bool)
        self.check[check_positions] = True
        self.inner = self.fit & ~self.check
        if not bool(self.held.any()) or not bool(self.inner.any()) or not bool(self.check.any()):
            raise SystemExit(
                f"the corpus split left an empty piece (held {int(self.held.sum())}, inner "
                f"{int(self.inner.sum())}, check {int(self.check.sum())}); the ridge needs all three")

    def _timed(self, name: str, fn, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        result = fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings[name] = round(self.timings.get(name, 0.0) + time.time() - started, 2)
        return result

    def stage1(self, k_pool: int):
        return self._timed("stage1_seconds", super().stage1, k_pool)

    def stage2(self, channels: torch.Tensor, k: int):
        return self._timed("stage2_seconds", super().stage2, channels, k)

    def refit(self, channels: torch.Tensor):
        """`CoresetFold.refit` with `measured_solve_gram` swapped in for the duration of the call.

        The lane file is untouched: the swap lives on the c14 module object only while
        this refit runs and is restored (and asserted restored) in ``finally``.
        """
        fit_rows = torch.nonzero(self.fit).flatten().to(self.device)
        fit_y_sq = 0.0
        for start in range(0, fit_rows.numel(), self.chunk):
            fit_y_sq += float(self.y_lat[fit_rows[start:start + self.chunk]].double().pow(2).sum())
        stash: dict[str, Any] = {}
        original = c14.law_abiding_solve_gram

        def swapped(*args, **kw):
            return measured_solve_gram(*args, fit_y_sq=fit_y_sq, stash=stash, **kw)

        c14.law_abiding_solve_gram = swapped
        try:
            w2p, bank, ladder = self._timed("refit_seconds", super().refit, channels)
        finally:
            c14.law_abiding_solve_gram = original
        assert c14.law_abiding_solve_gram is original
        self._last_prior = stash.get("prior")
        ladder["solver"] = f"{TOOL_NAME}.measured_solve_gram v{TOOL_VERSION} (c14.prior_anchored_gram arithmetic)"
        ladder["w2_bank_norm"] = float(bank.w2.float().norm())
        if ladder["w2_bank_norm"] > 0:
            ladder["prior_norm_over_w2_bank"] = float(ladder["prior_norm"] / ladder["w2_bank_norm"])
            for row in ladder["ladder"]:
                row["solution_norm_over_w2_bank"] = float(row["solution_norm"] / ladder["w2_bank_norm"])
        return w2p, bank, ladder

    # -- scoring: the lane's held-out score plus the baselines the gates need ----------
    def _prediction_chunks(self, bank, w2p: torch.Tensor, rows: torch.Tensor,
                           runtime_dtype: bool = False):
        for start in range(0, rows.numel(), self.chunk):
            piece = rows[start:start + self.chunk]
            psi = bank.psi(self.latent[piece], self.beta, self.linear_beta)
            if runtime_dtype:
                # what `dense_y_lat` / k3_layer's dense path does: W2' cast to the activation dtype
                pred = F.linear(psi, w2p.to(psi.device, psi.dtype)).float()
            else:
                pred = F.linear(psi.float(), w2p.to(psi.device).float())
            del psi
            yield pred, self.y_lat[piece]
            del pred

    def _routed_write(self, latent: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg["rms_norm_eps"])
        body = self.w.body
        norm_w = body["block_sparse_moe.routed_expert_norm.weight"]
        up_w = body["block_sparse_moe.routed_expert_up_proj.weight"]
        dtype = up_w.dtype
        return F.linear(rms_norm(latent.to(dtype), norm_w, eps), up_w)

    @torch.no_grad()
    def _score_map(self, bank, w2p: torch.Tensor, rows: torch.Tensor, runtime_dtype: bool = False) -> dict[str, Any]:
        latent_scores = WriteScoreAccumulator(self.chunk)
        write_scores_acc = WriteScoreAccumulator(self.chunk)
        for pred, target in self._prediction_chunks(bank, w2p, rows, runtime_dtype=runtime_dtype):
            latent_scores.update(pred, target)
            write_scores_acc.update(self._routed_write(pred), self._routed_write(target))
            del pred, target
        return {"y_lat": latent_scores.result(), "routed_write": write_scores_acc.result()}

    def _score_extras(self, bank, w2p: torch.Tensor) -> dict[str, Any]:
        held = torch.nonzero(self.held).flatten().to(self.device)
        fit = torch.nonzero(self.fit).flatten().to(self.device)
        out: dict[str, Any] = {}
        dtype_name = str(self.latent.dtype).replace("torch.", "")
        runtime = self._score_map(bank, w2p, held, runtime_dtype=True)
        out["y_lat_heldout_runtime_dtype"] = {**runtime["y_lat"], "dtype": dtype_name}
        out["routed_write_heldout_runtime_dtype"] = {**runtime["routed_write"], "dtype": dtype_name}
        on_fit = self._score_map(bank, w2p, fit)
        out["y_lat_fit"] = on_fit["y_lat"]
        out["routed_write_fit"] = on_fit["routed_write"]
        zero = self._score_map(bank, torch.zeros_like(w2p), held)
        baselines: dict[str, Any] = {"zero": {"y_lat_heldout": zero["y_lat"], "routed_write_heldout": zero["routed_write"]}}
        prior = getattr(self, "_last_prior", None)
        if prior is not None:
            scored = self._score_map(bank, prior.to(w2p.dtype), held)
            baselines["prior"] = {"y_lat_heldout": scored["y_lat"], "routed_write_heldout": scored["routed_write"],
                                  "what": "the parent's W2 column x the channel's mean router mass (c14's ridge prior)"}
        out["baselines"] = baselines
        out["rows"] = {"held": int(held.numel()), "fit": int(fit.numel())}
        out["scoring"] = {"version": "k3.chunked-write-scores.v1", "chunk_rows": self.chunk,
                          "rows_sampled": False, "norm_reduction": "FP64 squared norms over FP32 values",
                          "cosine": "FP32 per row; every scalar retained on CPU for full mean and exact linear p05"}
        out["what"] = ("y_lat_heldout / routed_write_heldout: the lane's score (fp32 W2'); *_runtime_dtype: W2' "
                       "cast as the dense path casts it; y_lat_fit: the installed map on the fit rows (G2); "
                       "baselines: the zero map and the prior map on the same held rows (G1)")
        return out

    def score(self, bank, w2p: torch.Tensor):
        # The inherited held path also concatenates the entire population. Use
        # the same streamed map for held, fit, runtime-dtype and baseline scores.
        held = torch.nonzero(self.held).flatten().to(self.device)
        scored = self._timed("score_seconds", self._score_map, bank, w2p, held)
        scores = {"y_lat_heldout": scored["y_lat"], "routed_write_heldout": scored["routed_write"]}
        scores.update(self._timed("score_extras_seconds", self._score_extras, bank, w2p))
        return scores


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
class DualStreamRun(StreamRun):
    """`StreamRun` carrying TWO residual pairs, fitting the folded one layer by layer."""

    def __init__(self, args, index: K3Index, corpus: Corpus, stager: ShardStager, policy: dict[str, Any]):
        # StreamRun.__init__ is not called: it makes a cache tree and reads a
        # single-stream receipt.  What its methods need is set here.
        self.args = args
        self.index = index
        self.cfg = index.text_config
        self.device = canonical_device(args.device)
        self.experts_device = canonical_device(args.experts_device or args.device)
        self.resid_device = canonical_device(args.resid_device)
        self.compute_dtype = torch_dtype(args.compute_dtype)
        self.corpus = corpus
        self.stager = stager
        self.policy = policy
        self.out = os.path.abspath(args.out)
        self.receipt_dir = os.path.abspath(args.receipt_dir)
        self.run_id = stager.run_id
        self.block_size = int(self.cfg["attn_res_block_size"] or 0)
        if self.block_size <= 0:
            raise SystemExit("this pass only implements the attn-residual variant (attn_res_block_size set)")
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.receipt_dir, exist_ok=True)
        self._head_cache: dict[str, torch.Tensor] | None = None
        self.installed: dict[int, str] = {}  # MoE layer -> sha256 of the ffn.safetensors the folded pair carries
        self.snapshot_manifest_path = os.path.join(self.out, "snapshots", SNAPSHOT_MANIFEST)
        self.snapshot_manifest = self._load_snapshot_manifest()
        self.receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "tool_sha256": sha256_file(os.path.abspath(__file__)),
            "lane_dir": LANE_DIR,
            "lane_sha256": lane_file_digests(),
            "run_id": self.run_id,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arguments": {k: v for k, v in vars(args).items()},
            "model": self._model_identity(),
            "share_policy": policy,
            "corpus": corpus.record(),
            "device": str(self.device),
            "experts_device": str(self.experts_device),
            "resid_device": str(self.resid_device),
            "compute_dtype": str(self.compute_dtype) if self.compute_dtype else "parent (bf16)",
            "kda_backend": args.kda_backend,
            "devices": device_inventory(),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "fit": {
                "input": "FOLDED pair: latent_in = W_down . post-attention residual of the folded stream",
                "target": "TEACHER pair: y_lat, the PRE-norm fp32 routed accumulator (routed output)",
                "selection_mass": "teacher router top-k on the teacher's own input",
                "split": "corpus in-band fit_mask (fit rows) / held_mask (held rows); ridge check split = "
                         "every third contiguous block of the fit positions",
                "rung": int(args.rung), "k_pool_factor": int(args.k_pool_factor),
                "zero_gain_rule": "section 3b: no division by a norm gain anywhere; zero gains are counted",
            },
            "snapshots": {"every": int(args.snapshot_every), "keep": int(args.snapshot_keep), "written": []},
            "layers": {},
        }
        self.resume_context = self._resume_context()
        self.receipt["resume_context"] = self.resume_context
        self._carry_forward_previous_receipt()

    # -- identity ------------------------------------------------------------
    def _resume_context(self) -> dict[str, Any]:
        """Bind cached acceptance to what produced it, without rehashing the 1.56 TB parent.

        Source file size/mtime is an inexpensive change detector, not proof of
        unchanged shard bytes. Full shard identity remains the separate digest gate.
        Operational flags (staging, retention, layer window, acceptance mode) do
        not change this key; numerical/scoring inputs deliberately do.
        """
        model = self.receipt["model"]
        source_files = {}
        for name in sorted(set(self.index.weight_map.values())):
            stat = os.stat(os.path.join(self.args.src_dir, name))
            source_files[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        numerical_args = ("compute_dtype", "kda_backend", "k_pool_factor", "row_chunk", "token_chunk",
                          "chunk", "holdout_blocks", "divergence_rows", "review_topk")
        packages = {}
        for name in ("flash-linear-attention", "triton"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        matmul = torch.backends.cuda.matmul
        backend_flags = {key: getattr(matmul, key, None) for key in (
            "allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")}
        for name in ("flash_sdp_enabled", "math_sdp_enabled", "mem_efficient_sdp_enabled", "cudnn_sdp_enabled"):
            getter = getattr(torch.backends.cuda, name, None)
            backend_flags[name] = getter() if getter is not None else None
        backend_flags.update({"cudnn_" + name: getattr(torch.backends.cudnn, name, None)
                              for name in ("enabled", "benchmark", "deterministic", "allow_tf32")})
        return {
            "version": RESUME_CONTEXT_VERSION,
            "model": {"source_directory": os.path.normcase(os.path.realpath(self.args.src_dir)),
                      "config_sha256": model["config_sha256"], "index_sha256": model["index_sha256"],
                      "source_file_metadata": source_files},
            "source_identity_scope": "config/index hashes and shard size/mtime change detection; "
                                     "not a full shard digest verification",
            "corpus": self.corpus.sha256,
            "runtime": {**{key: getattr(self.args, key) for key in numerical_args},
                        "device": str(self.device), "experts_device": str(self.experts_device),
                        "resid_device": str(self.resid_device), "devices": self.receipt["devices"],
                        "float32_matmul_precision": torch.get_float32_matmul_precision(),
                        "num_threads": torch.get_num_threads(), "backend_flags": backend_flags},
            "software": {"tool_sha256": self.receipt["tool_sha256"],
                         "lane_sha256": self.receipt["lane_sha256"], "torch": str(torch.__version__),
                         "cuda": torch.version.cuda, "packages": packages},
        }

    def _model_identity(self) -> dict[str, Any]:
        src = os.path.abspath(self.args.src_dir)
        index_path = os.path.join(src, "model.safetensors.index.json")
        config_path = os.path.join(src, "config.json")
        return {
            "model_dir": os.path.abspath(self.args.model_dir), "src_dir": src,
            "index_sha256": sha256_file(index_path), "config_sha256": sha256_file(config_path),
            "num_layers": self.index.num_layers, "hidden_size": self.index.hidden_size,
            "attn_res_block_size": self.block_size,
            "layer_prefix": self.index.layer_prefix,
            "digest_verify_receipt": self._digest_receipt_status(),
        }

    def _digest_receipt_status(self) -> dict[str, Any]:
        path = self.args.digest_receipt
        if not path or not os.path.exists(path):
            return {"path": path, "present": False,
                    "note": "the Lexar copy is treated as size-verified until its digest receipt lands"}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                got = json.load(handle)
        except (OSError, ValueError) as exc:
            return {"path": path, "present": True, "error": repr(exc)[:200]}
        return digest_receipt_status(got, path, expected_files=len(set(self.index.weight_map.values())))

    def _carry_forward_previous_receipt(self) -> None:
        """On --resume, carry the NEWER of the full / partial receipt (by mtime), and only
        the layer entries that still describe what is on disk at THIS rung.

        A carried MoE entry must have a done marker that validates (corpus, rung,
        artefact sha); entries without one are dropped and listed, so a receipt can
        never mix folds fitted at another rung with this run's.
        """
        if not self.args.resume:
            return
        candidates = [os.path.join(self.receipt_dir, n) for n in (RECEIPT_NAME, PARTIAL_RECEIPT_NAME)]
        candidates = [p for p in candidates if os.path.exists(p)]
        if not candidates:
            return
        candidates.sort(key=os.path.getmtime, reverse=True)
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    prior = json.load(handle)
            except (OSError, ValueError) as exc:
                self.receipt["resumed_from_receipt"] = {"path": path, "error": repr(exc)[:200]}
                continue
            if not isinstance(prior, dict) or prior.get("resume_context") != self.resume_context:
                self.receipt["resumed_from_receipt"] = {
                    "path": path, "carried": False,
                    "reason": "source/runtime/corpus resume context differs or is missing; replay/refit required"}
                continue
            if prior.get("corpus", {}).get("corpus_sha256") != self.corpus.sha256:
                self.receipt["resumed_from_receipt"] = {
                    "path": path, "carried": False,
                    "reason": "the previous receipt was written for a different corpus"}
                continue
            prior_rung = prior.get("arguments", {}).get("rung")
            if not isinstance(prior_rung, int) or isinstance(prior_rung, bool) or prior_rung != int(self.args.rung):
                self.receipt["resumed_from_receipt"] = {
                    "path": path, "carried": False, "previous_rung": prior_rung, "rung": int(self.args.rung),
                    "reason": "the previous receipt was written at another rung; its layers are not carried"}
                continue
            carried, dropped = {}, {}
            previous_layers = prior.get("layers", {})
            valid_markers = self.valid_done_chain(max((int(k) for k in previous_layers), default=-1))
            for key, entry in previous_layers.items():
                if entry.get("moe"):
                    marker = valid_markers.get(int(key))
                    ffn_sha = (entry.get("ffn") or {}).get("sha256")
                    if (marker is None or marker.get("artefact_sha256") != ffn_sha
                            or entry.get("prefix_lineage") != marker.get("prefix_lineage")):
                        dropped[key] = ("no done marker validates this layer's ffn and receipt at this "
                                        "corpus/rung/sha/prefix lineage/acceptance")
                        continue
                carried[key] = entry
            self.receipt["layers"].update(carried)
            self.receipt["snapshots"]["written"] = list(prior.get("snapshots", {}).get("written", []))
            sidecar = os.path.join(self.receipt_dir, RECEIPT_SHA_SIDECAR)
            previous_sha = prior.get("receipt_sha256")
            if previous_sha is None and os.path.basename(path) == RECEIPT_NAME and os.path.exists(sidecar):
                with open(sidecar, "r", encoding="utf-8") as handle:
                    words = handle.read().split()
                previous_sha = words[0] if words else None
            self.receipt["resumed_from_receipt"] = {
                "path": path, "carried": True, "chosen_by": "newest mtime of the full / partial receipt",
                "layers_carried_forward": sorted(int(k) for k in carried),
                "layers_dropped": dropped,
                "previous_receipt_sha256": previous_sha,
                "previous_tool_version": prior.get("tool_version"),
            }
            return

    # -- model-level reads: from the SOURCE, never through the stager ----------
    def _model_tensor(self, name_candidates: list[str]) -> torch.Tensor:
        """Embedding / head tensors are read off the source shard, as `StreamRun` does.

        They are needed twice per run (the embedding at the start, the head at the
        end) and belong to no layer's shard plan; staging them parked their shard
        in the window for the whole pass, un-releasable, which at
        ``--stage-window 1`` deadlocked the worker.  Each read is recorded in the
        staging events as ``model_level_read_from_source``.
        """
        for name in name_candidates:
            if name in self.index.weight_map:
                shard = self.index.weight_map[name]
                got = read_tensors(os.path.join(self.args.src_dir, shard), [name])
                self.stager.note_source_read(shard, name)
                return got[name]
        raise SystemExit(f"none of {name_candidates} is in the index")

    # -- snapshots (both pairs, provenance-stamped) ---------------------------
    def snapshot_path(self, layer_idx: int, stream: str = "teacher") -> str:
        suffix = "" if stream == "teacher" else "_folded"
        return os.path.join(self.out, "snapshots", f"before_L{layer_idx:02d}{suffix}.safetensors")

    def _load_snapshot_manifest(self) -> dict[str, Any]:
        if os.path.exists(self.snapshot_manifest_path):
            with open(self.snapshot_manifest_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {"tool": TOOL_NAME, "files": {}}

    def _write_snapshot_manifest(self) -> None:
        write_receipt(self.snapshot_manifest_path, self.snapshot_manifest)

    def installed_stamp(self) -> str:
        """Canonical JSON of {layer: ffn artefact sha256} for every fold the folded pair carries."""
        return canonical_json({str(k): v for k, v in sorted(self.installed.items())})

    def prefix_lineage(self, layer_idx: int, installed: dict[int, str] | None = None) -> dict[str, Any]:
        """The complete fold ancestry of this input, excluding the current layer."""
        installed = self.installed if installed is None else installed
        return {"version": PREFIX_LINEAGE_VERSION,
                "upstream_ffn_sha256": {str(k): v for k, v in sorted(installed.items()) if k < layer_idx}}

    def prior_moe_layers(self, layer_idx: int) -> set[int]:
        return {i for i in range(layer_idx) if self.index.is_moe_layer(i)}

    def save_dual_snapshot(self, layer_idx: int, teacher: Pair, folded: Pair) -> dict[str, Any]:
        """Both pairs, stamped with the corpus, the rung, the tool version and the sha of every
        installed ffn.safetensors -- what a resume must find unchanged on disk."""
        stamp = {"input_ids_sha256": self.corpus.sha256, "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
                 "run_id": self.run_id, "rung": str(int(self.args.rung)),
                 "resume_context": canonical_json(self.resume_context),
                 "prefix_lineage_version": PREFIX_LINEAGE_VERSION,
                 "installed_ffn_sha256": self.installed_stamp(),
                 "installed_layers": str(len(self.installed))}
        record = {"layer": layer_idx, "corpus_sha256": self.corpus.sha256, "rung": int(self.args.rung),
                  "installed_layers": sorted(self.installed), "files": {}}
        for stream, pair in (("teacher", teacher), ("folded", folded)):
            path = self.snapshot_path(layer_idx, stream)
            digest = save_tensors(path, {"prefix_sum": pair.prefix.cpu(), "block_residual": pair.blocks.cpu()},
                                  metadata={"layer": str(layer_idx), "stream": stream,
                                            "blocks": str(int(pair.blocks.shape[1])), **stamp})
            record["files"][stream] = {"path": path, "sha256": digest,
                                       "bytes": os.path.getsize(path)}
            self.snapshot_manifest["files"][path] = {"layer": layer_idx, "stream": stream, "sha256": digest,
                                                     "corpus_sha256": self.corpus.sha256,
                                                     "rung": int(self.args.rung),
                                                     "installed": dict(sorted(self.installed.items())),
                                                     "run_id": self.run_id}
        self._write_snapshot_manifest()
        self.receipt["snapshots"]["written"].append(record)
        self._prune_snapshots(keep=int(self.args.snapshot_keep))
        return record

    def snapshot_provenance(self, layer_idx: int) -> tuple[bool, str, dict[str, Any]]:
        """(usable, reason, stamp) for the dual snapshot before ``layer_idx``.

        Usable means: both files exist, stamped with this corpus, and every fold the
        stamp says is installed has a done marker that validates (corpus, rung,
        artefact sha on disk) with the SAME sha the snapshot was written under.
        A snapshot carrying folds is refused at another rung.
        """
        stamps = {}
        for stream in ("teacher", "folded"):
            path = self.snapshot_path(layer_idx, stream)
            if not os.path.exists(path):
                return False, f"{path} is missing", {}
            stamp = read_metadata(path)
            if stamp.get("resume_context") != canonical_json(self.resume_context):
                return False, f"snapshot {path} carries a different or missing source/runtime/corpus context", stamp
            if stamp.get("input_ids_sha256") != self.corpus.sha256:
                return False, (f"snapshot {path} was written for input_ids {stamp.get('input_ids_sha256')!r}, "
                               f"this run streams {self.corpus.sha256!r}"), stamp
            if stamp.get("prefix_lineage_version") != PREFIX_LINEAGE_VERSION:
                return False, f"snapshot {path} carries no current prefix-lineage version (older tool)", stamp
            stamps[stream] = stamp
        stamp = stamps["folded"]
        installed_raw = stamp.get("installed_ffn_sha256")
        if installed_raw is None:
            return False, f"snapshot before layer {layer_idx} carries no installed-fold stamp (older tool)", stamp
        try:
            raw = json.loads(installed_raw)
            if not isinstance(raw, dict):
                raise ValueError("installed folds must be an object")
            installed = {int(k): v for k, v in raw.items()}
            if any(str(int(k)) != k or not isinstance(v, str) or len(v) != 64
                   or any(c not in "0123456789abcdef" for c in v) for k, v in raw.items()):
                raise ValueError("installed folds must map canonical layer indices to SHA256 digests")
        except (TypeError, ValueError):
            return False, f"snapshot before layer {layer_idx} carries malformed installed-fold lineage", stamp
        if set(installed) != self.prior_moe_layers(layer_idx):
            return False, f"snapshot before layer {layer_idx} does not carry the complete prior MoE prefix", stamp
        if stamps["teacher"].get("installed_ffn_sha256") != installed_raw:
            return False, "teacher and folded snapshots disagree on the installed folds", stamp
        if installed and any(item.get("rung") != str(int(self.args.rung)) for item in stamps.values()):
            return False, (f"snapshot before layer {layer_idx} carries {len(installed)} folds fitted at rung "
                           f"{stamp.get('rung')!r} (teacher {stamps['teacher'].get('rung')!r}), "
                           f"this run is --rung {self.args.rung}"), stamp
        validated_prefix = {}
        for fold_layer, sha in sorted(installed.items()):
            marker = self.read_done_marker(fold_layer, installed=validated_prefix)
            if marker is None or marker.get("artefact_sha256") != sha:
                return False, (f"snapshot before layer {layer_idx} was written with L{fold_layer:02d}/ffn "
                               f"sha {sha[:12]}..., but no done marker validates that file now"), stamp
            validated_prefix[fold_layer] = sha
        return True, "ok", {**stamp, "installed": installed}

    def _prune_snapshots(self, keep: int) -> None:
        """Keep the newest `keep` layers' pairs written by THIS TOOL; delete nothing else."""
        if keep <= 0:
            return
        layers = sorted({rec["layer"] for rec in self.snapshot_manifest["files"].values()})
        for layer in layers[:-keep] if len(layers) > keep else []:
            for path, rec in list(self.snapshot_manifest["files"].items()):
                if rec["layer"] != layer:
                    continue
                if os.path.exists(path):
                    os.remove(path)
                del self.snapshot_manifest["files"][path]
                self.receipt["snapshots"].setdefault("pruned", []).append(
                    {"path": path, "layer": layer, "stream": rec["stream"], "sha256": rec["sha256"]})
        self._write_snapshot_manifest()

    def dual_snapshot_available(self, layer_idx: int) -> bool:
        usable, reason, _ = self.snapshot_provenance(layer_idx)
        if not usable and os.path.exists(self.snapshot_path(layer_idx, "teacher")):
            self.receipt.setdefault("snapshots_refused", []).append({"layer": layer_idx, "reason": reason})
        return usable

    def load_dual_snapshot(self, layer_idx: int) -> tuple[Pair, Pair, dict[str, Any]]:
        from safetensors.torch import load_file

        usable, reason, stamp = self.snapshot_provenance(layer_idx)
        if not usable:
            raise SystemExit(f"refusing to resume from the dual snapshot before layer {layer_idx}: {reason}")
        pairs = {}
        record: dict[str, Any] = {"layer": layer_idx, "files": {},
                                  "installed_layers": sorted(stamp["installed"]), "rung": stamp.get("rung")}
        for stream in ("teacher", "folded"):
            path = self.snapshot_path(layer_idx, stream)
            got = load_file(path)
            prefix = got["prefix_sum"].to(self.resid_device)
            blocks = got["block_residual"].to(self.resid_device)
            if self.compute_dtype is not None:
                prefix, blocks = prefix.to(self.compute_dtype), blocks.to(self.compute_dtype)
            pairs[stream] = Pair(prefix, blocks)
            record["files"][stream] = {"path": path, "sha256": sha256_file(path), "stamp": read_metadata(path)}
        self.installed = dict(stamp["installed"])
        return pairs["teacher"], pairs["folded"], record

    def resume_point(self, layers: list[int]) -> int | None:
        available = [i for i in layers if self.dual_snapshot_available(i)]
        return max(available) if available else None

    # -- done markers -------------------------------------------------------
    def layer_dir(self, layer_idx: int) -> str:
        return os.path.join(self.out, "dense", f"L{layer_idx:02d}")

    def done_marker_path(self, layer_idx: int) -> str:
        return os.path.join(self.layer_dir(layer_idx), "done.json")

    def valid_done_chain(self, through_layer: int) -> dict[int, dict[str, Any]]:
        """Validate in order, hashing each artifact once; a broken ancestor ends the chain."""
        valid, installed = {}, {}
        for layer in range(through_layer + 1):
            if not self.index.is_moe_layer(layer):
                continue
            marker = self.read_done_marker(layer, installed=installed)
            if marker is None:
                break
            valid[layer] = marker
            installed[layer] = marker["artefact_sha256"]
        return valid

    def read_done_marker(self, layer_idx: int, *, installed: dict[int, str] | None = None) -> dict[str, Any] | None:
        """A reusable marker must match the actual prefix and this run's gate eligibility.

        Callers with an already-validated carried prefix pass it explicitly. Other
        callers validate the on-disk chain once, rather than recursively rehashing it.
        Legacy MoE markers without an explicit lineage are not reusable, even at L01.
        """
        if installed is None and self.index.is_moe_layer(layer_idx):
            return self.valid_done_chain(layer_idx).get(layer_idx)
        path = self.done_marker_path(layer_idx)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(marker, dict):
            return None
        if marker.get("resume_context") != self.resume_context:
            return None
        artefact = marker.get("artefact")
        if (marker.get("corpus_sha256") != self.corpus.sha256 or not isinstance(artefact, str)
                or not artefact or not os.path.exists(artefact)):
            return None
        if self.index.is_moe_layer(layer_idx):
            rung = marker.get("rung")
            if not isinstance(rung, int) or isinstance(rung, bool) or rung != int(self.args.rung):
                return None
        if sha256_file(artefact) != marker.get("artefact_sha256"):
            raise SystemExit(f"done marker {path} names {artefact} with sha {marker.get('artefact_sha256')}, "
                             "but the file on disk hashes differently; refusing to trust it")
        if self.index.is_moe_layer(layer_idx):
            if (set(installed) != self.prior_moe_layers(layer_idx)
                    or marker.get("prefix_lineage") != self.prefix_lineage(layer_idx, installed)):
                return None
            if self.args.acceptance == "gate":
                metrics = marker.get("fold_metrics")
                acceptance = metrics.get("acceptance") if isinstance(metrics, dict) else None
                if (not isinstance(acceptance, dict) or acceptance.get("version") != ACCEPTANCE_VERSION
                        or acceptance.get("constants") != ACCEPTANCE_CONSTANTS
                        or acceptance.get("verdict") != "accepted"):
                    return None
        return marker

    def write_done_marker(self, layer_idx: int, artefact: str, digest: str, extra: dict[str, Any]) -> str:
        marker = {"layer": layer_idx, "artefact": artefact, "artefact_sha256": digest,
                  "corpus_sha256": self.corpus.sha256, "rung": int(self.args.rung) if extra.get("moe") else None,
                  "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "run_id": self.run_id,
                  "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
        marker["resume_context"] = self.resume_context
        if extra.get("moe"):
            marker["prefix_lineage"] = self.prefix_lineage(layer_idx)
        return write_receipt(self.done_marker_path(layer_idx), marker)

    # -- embedding -----------------------------------------------------------
    def initial_pair(self) -> Pair:
        pieces = []
        for seg in self.corpus.segments:
            embedded = self.embed(seg.ids)  # [rows, T, H] on self.device (StreamRun.embed)
            pieces.append(embedded.reshape(-1, embedded.shape[-1]).to(self.resid_device))
            del embedded
        prefix = torch.cat(pieces, dim=0)
        if self.compute_dtype is not None:
            prefix = prefix.to(self.compute_dtype)
        blocks = prefix.new_zeros(prefix.shape[0], 0, prefix.shape[1])
        return Pair(prefix, blocks)

    # -- the head, per row chunk ----------------------------------------------
    def _head_tensors(self) -> dict[str, torch.Tensor]:
        if self._head_cache is None:
            prefix = self.index.model_prefix
            lm_prefix = self.index.layer_prefix.split("model.layers.")[0]
            got = {
                "proj": self._model_tensor([f"{prefix}output_attn_res_proj.weight"]),
                "norm": self._model_tensor([f"{prefix}output_attn_res_norm.weight"]),
                "final": self._model_tensor([f"{prefix}norm.weight"]),
                "lm_head": self._model_tensor([f"{lm_prefix}lm_head.weight", "lm_head.weight"]),
            }
            for key, tensor in got.items():
                tensor = tensor.to(self.device)
                if self.compute_dtype is not None:
                    tensor = tensor.to(self.compute_dtype)
                got[key] = tensor
            self._head_cache = got
        return self._head_cache

    def logits_rows(self, pair: Pair, start: int, stop: int, n_rows: int, seq: int) -> torch.Tensor:
        """`StreamRun.head` on one row chunk: attn-res read -> final norm -> lm_head, fp32 out."""
        head = self._head_tensors()
        eps = float(self.cfg["rms_norm_eps"])
        prefix = pair.prefix[start:stop].to(self.device)
        blocks = pair.blocks[start:stop].to(self.device)
        hidden = apply_attn_res(prefix, blocks, head["proj"], head["norm"], eps,
                                token_chunk_size=int(self.args.token_chunk))
        hidden = rms_norm(hidden, head["final"], eps)
        return F.linear(hidden, head["lm_head"]).float().view(n_rows, seq, -1)

    # -- the pass -------------------------------------------------------------
    def require_peak_rss(self, where: str, layer_idx: int | None = None) -> int:
        """Require an actual process-lifetime peak measurement; never report unknown as zero."""
        peak = peak_rss_bytes()
        available = peak is not None and peak > 0
        record = {"required": True, "measurement_available": available,
                  "peak_rss_bytes": int(peak) if available else None,
                  "source": peak_rss_source(), "where": where,
                  "scope": "whole-process lifetime peak, not an isolated per-layer peak"}
        self.receipt.setdefault("resources", {})["host_peak_rss"] = record
        if not available:
            message = f"process peak RSS measurement unavailable ({where}); refusing unmeasured completion"
            detail = {"gate": "host_peak_rss_measurement", "host_peak_rss": record}
            self.receipt["peak_rss_bytes"] = None
            self.receipt["refusal"] = {"layer": layer_idx, "message": message,
                                       "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **detail}
            write_receipt(os.path.join(self.receipt_dir, PARTIAL_RECEIPT_NAME), self.receipt)
            raise Refusal(message, detail)
        return int(peak)

    def run(self) -> dict[str, Any]:
        args = self.args
        self.require_peak_rss("before_pass")
        layers = parse_layers(args.layers, self.index.num_layers)
        if layers != list(range(layers[0], layers[-1] + 1)):
            raise SystemExit(f"--layers must be one contiguous range for a streamed pass, got {args.layers}")
        first_moe = next((i for i in range(self.index.num_layers) if self.index.is_moe_layer(i)),
                         self.index.num_layers)
        start_layer = layers[0]
        resumed: dict[str, Any] | None = None
        auto_prefix: list[int] = []
        if args.resume:
            found = self.resume_point(layers)
            if found is not None and found >= start_layer:
                start_layer = found
        if start_layer == layers[0] and start_layer == 0:
            teacher = self.initial_pair()
            folded = teacher.clone()
            folded_is_teacher = True
            self.receipt["start"] = {"layer": 0, "from": "embedding"}
        elif (start_layer == layers[0] and not self.dual_snapshot_available(start_layer)
              and all(not self.index.is_moe_layer(i) for i in range(start_layer))):
            # Every layer below the window is dense (K3: only layer 0), so the folded pair
            # IS the teacher pair there and the prefix is recomputed from the embedding
            # instead of refusing -- the S2P `--layers 1-1` command runs without a prior
            # window.  Those layers are processed (L00/mlp.safetensors written verbatim)
            # and listed under start.auto_prefix_layers.
            auto_prefix = list(range(0, start_layer))
            teacher = self.initial_pair()
            folded = teacher.clone()
            folded_is_teacher = True
            self.receipt["start"] = {"layer": 0, "from": "embedding", "requested_first_layer": start_layer,
                                     "auto_prefix_layers": auto_prefix,
                                     "note": "no dual snapshot before the window; every earlier layer is dense, "
                                             "so the pair was rebuilt from the embedding through them"}
            start_layer = 0
        else:
            if not self.dual_snapshot_available(start_layer):
                raise SystemExit(
                    f"no dual snapshot (teacher + folded, stamped with this corpus at this rung) exists before "
                    f"layer {start_layer} under {os.path.join(self.out, 'snapshots')}; a window that does not "
                    "start at layer 0 needs --resume and the end-of-window snapshot of the previous window "
                    f"(refused: {self.receipt.get('snapshots_refused')})")
            teacher, folded, resumed = self.load_dual_snapshot(start_layer)
            folded_is_teacher = False
            self.receipt["start"] = {"layer": start_layer, "from": "dual_snapshot", **resumed}
            log(f"resumed both pairs from the snapshot before layer {start_layer} "
                f"(installed folds: {sorted(self.installed)})")
        self.receipt["first_moe_layer"] = first_moe
        if args.body_dir and self.receipt["start"]["from"] == "embedding":
            self.receipt["body_model_level"] = self.write_model_level_verbatim()

        wall_start = time.time()
        processed = auto_prefix + [i for i in layers if i >= start_layer]
        plan_shards = {i: self.index.shards_for(self.index.layer_keys(i)) for i in processed}
        partial_path = os.path.join(self.receipt_dir, PARTIAL_RECEIPT_NAME)
        for layer_idx in processed:
            just_loaded = resumed is not None and layer_idx == start_layer
            if args.snapshot_every and layer_idx % args.snapshot_every == 0 and not just_loaded:
                self.save_dual_snapshot(layer_idx, teacher, folded)
            upcoming = [s for j in processed if j > layer_idx for s in plan_shards[j]]
            # a layer that straddles more shards than the window holds would wait on itself
            self.stager.window = max(self.stager.window, len(plan_shards[layer_idx]))
            self.stager.queue(plan_shards[layer_idx] + upcoming[: self.stager.prefetch_n])
            try:
                entry = self.run_layer(layer_idx, teacher, folded, folded_is_teacher)
            except SystemExit as exc:  # a named refusal: recorded in the partial receipt, then raised
                self.receipt["refusal"] = {"layer": layer_idx, "message": str(exc),
                                           "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           **getattr(exc, "detail", {})}
                self.receipt["staging"] = self.stager.record()
                write_receipt(partial_path, self.receipt)
                raise
            teacher, folded = entry.pop("_teacher"), entry.pop("_folded")
            folded_is_teacher = entry.pop("_folded_is_teacher")
            still_needed = {s for j in processed if j > layer_idx for s in plan_shards[j]}
            entry["staged_release"] = self.stager.release(plan_shards[layer_idx], still_needed)
            self.receipt["layers"][str(layer_idx)] = entry
            self.receipt["staging"] = self.stager.record()
            self.receipt["peak_rss_bytes_so_far"] = self.require_peak_rss("after_layer", layer_idx)
            write_receipt(partial_path, self.receipt)
            log(f"L{layer_idx:02d} done in {entry['layer_seconds']}s: read {entry['read']['gigabytes']} GB at "
                f"{entry['read']['MB_per_s']} MB/s; folded-vs-teacher prefix relL2 (held) "
                f"{entry['stream_divergence']['held']['relL2']:.3e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if processed and args.snapshot_every:
            nxt = processed[-1] + 1
            if nxt < self.index.num_layers:
                self.receipt["end_of_window_snapshot"] = self.save_dual_snapshot(nxt, teacher, folded)

        if layers[-1] == self.index.num_layers - 1:
            self.receipt["end_of_stack"] = self.end_of_stack(teacher, folded)
            self.receipt["window_acceptance"] = self.window_acceptance(self.receipt["end_of_stack"])
            wa = self.receipt["window_acceptance"]
            log(f"G4 window acceptance: {wa['verdict']} (held KL/fitted layer {wa.get('kl_per_fitted_layer')}, "
                f"top-1 {wa.get('held_top1_agreement')})")
            if wa["verdict"] == "refused" and wa.get("enforced"):
                self.receipt["refusal"] = {"layer": layers[-1], "gate": "G4_window_intelligence",
                                           "message": wa["message"],
                                           "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           "gates": wa["gates"]}
                write_receipt(partial_path, self.receipt)
                raise Refusal(wa["message"], {"gate": "G4_window_intelligence", "gates": wa["gates"]})
        else:
            self.receipt["window_acceptance"] = {
                "version": ACCEPTANCE_VERSION, "verdict": "not_evaluated",
                "note": "G4 reads end_of_stack (held top-1 / KL), which exists only when the window ends at "
                        "the last layer; per-layer G1-G3 gated every fitted layer of this window"}

        self.receipt["final_prefix_sha256"] = {"teacher": sha256_tensor(teacher.prefix),
                                               "folded": sha256_tensor(folded.prefix)}
        self.receipt["final_block_stack_blocks"] = int(teacher.blocks.shape[1])
        self.receipt["layers_processed_this_run"] = processed
        self.receipt["layers_folded"] = sorted(int(k) for k, v in self.receipt["layers"].items() if v.get("moe"))
        self.receipt["installed_ffn_sha256"] = {str(k): v for k, v in sorted(self.installed.items())}
        self.receipt["total_seconds"] = round(time.time() - wall_start, 1)
        self.receipt["peak_rss_bytes"] = self.require_peak_rss("completion")
        self.receipt["staging"] = self.stager.record()
        self.stager.close()
        self.receipt["determinism_sha256"] = self.determinism_sha256()
        path = os.path.join(self.receipt_dir, RECEIPT_NAME)
        digest = write_receipt(path, self.receipt)
        sidecar = os.path.join(self.receipt_dir, RECEIPT_SHA_SIDECAR)
        with open(sidecar, "w", encoding="utf-8") as handle:
            handle.write(f"{digest}  {RECEIPT_NAME}\n")
        # a file cannot carry its own digest: the sha of the full receipt lives in the
        # sidecar and in the partial (written last, so it is the newer of the two)
        self.receipt["receipt_sha256"] = digest
        self.receipt["receipt_sha256_of"] = {"file": path, "sidecar": sidecar,
                                             "note": f"sha256 of {RECEIPT_NAME} as written; not inside that file"}
        write_receipt(partial_path, self.receipt)
        log(f"receipt -> {path} (sha256 {digest[:12]}..., sidecar {sidecar})")
        return self.receipt

    # -- one layer --------------------------------------------------------------
    def run_layer(self, layer_idx: int, teacher: Pair, folded: Pair, folded_is_teacher: bool) -> dict[str, Any]:
        args = self.args
        cfg = self.cfg
        layer_start = time.time()
        weights, raw_body, timing = read_layer(self.index, layer_idx, self.stager, self.device,
                                               self.experts_device, self.compute_dtype,
                                               progress_tag=f"L{layer_idx:02d}",
                                               keep_attention=bool(args.body_dir))
        runner = StreamedLayer(weights, cfg, kda_backend=args.kda_backend)
        entry: dict[str, Any] = {
            "layer": layer_idx, "is_kda": weights.is_kda, "moe": weights.is_moe, "read": timing,
            "parameters": weights.parameter_counts(),
            "layer_bytes_on_device": weights.device_bytes(self.device),
            "norm_gains": [zero_gain_census(name, weights.res[name])
                           for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                                        "self_attention_res_norm.weight", "mlp_res_norm.weight")
                           if name in weights.res],
            "finite_checks": [],
        }
        if args.body_dir:
            entry["body"] = self.write_body_verbatim(layer_idx, raw_body)
        if weights.is_moe:
            entry["norm_gains"].append(zero_gain_census(
                "block_sparse_moe.routed_expert_norm.weight",
                weights.body["block_sparse_moe.routed_expert_norm.weight"]))
        row_chunk, token_chunk = int(args.row_chunk), int(args.token_chunk)
        held_rows = torch.nonzero(self.corpus.held).flatten()
        held_dev = held_rows.to(self.device)

        # ---- teacher ---------------------------------------------------------
        compute_start = time.time()
        with torch.no_grad():
            t_carried, t_ffn_in, t_blocks_after = attention_half(
                runner, teacher, self.corpus, self.device, row_chunk=row_chunk, token_chunk=token_chunk)
            t_prefix_new, t_taps = ffn_half(runner, t_carried, t_ffn_in, cfg, device=self.device,
                                            token_chunk=token_chunk)
            t_held_moe_x = t_ffn_in[held_dev].cpu() if (args.held_rows and weights.is_moe) else None
            del t_carried
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        entry["teacher_compute_seconds"] = round(time.time() - compute_start, 2)
        entry["dequant_seconds"] = round(float(t_taps.get("dequant_seconds", 0.0)), 2)
        if "experts_touched" in t_taps:
            counts = t_taps["router_counts"]
            entry["router"] = {"experts": int(counts.numel()), "experts_touched": int(t_taps["experts_touched"]),
                               "max_tokens_on_one_expert": int(counts.max()) if counts.numel() else 0}
        new_teacher = Pair(t_prefix_new.to(self.resid_device), t_blocks_after)
        entry["teacher_temporary_lifetime"] = teacher_temporary_lifetime(t_prefix_new, new_teacher.prefix, t_taps)
        # The synchronous residual-device copy above owns the teacher output now.
        # Keeping this local GPU alias through folded MLA needlessly retains N*H.
        del t_prefix_new, t_ffn_in

        # ---- folded ----------------------------------------------------------
        fold_start = time.time()
        if not weights.is_moe:
            if folded_is_teacher:
                new_folded = new_teacher.clone()
                entry["folded"] = {"cloned_from_teacher": True,
                                   "note": "no MoE layer has been folded yet, so the folded pair IS the teacher pair"}
                f_prefix_new = new_folded.prefix
            else:
                with torch.no_grad():
                    f_carried, f_ffn_in, f_blocks_after = attention_half(
                        runner, folded, self.corpus, self.device, row_chunk=row_chunk, token_chunk=token_chunk)
                    f_prefix_new, _ = ffn_half(runner, f_carried, f_ffn_in, cfg, device=self.device,
                                               token_chunk=token_chunk)
                    del f_carried, f_ffn_in
                new_folded = Pair(f_prefix_new.to(self.resid_device), f_blocks_after)
                entry["folded"] = {"cloned_from_teacher": False, "dense_layer_recomputed": True}
            entry.update(self.write_dense_layer_verbatim(layer_idx, weights, raw_body))
            entry["fit_skipped"] = None
        else:
            with torch.no_grad():
                f_carried, f_ffn_in, f_blocks_after = attention_half(
                    runner, folded, self.corpus, self.device, row_chunk=row_chunk, token_chunk=token_chunk)
                f_latent = _chunked_linear(f_ffn_in, weights.body["block_sparse_moe.routed_expert_down_proj.weight"],
                                           token_chunk)
                f_held_moe_x = f_ffn_in[held_dev].cpu() if args.held_rows else None
            # section 3b: what reaches the fp64 Gram must be finite -- checked before the fold
            entry["finite_checks"].append(require_finite("folded latent_in (fit input)", f_latent, layer_idx))
            entry["finite_checks"].append(require_finite("teacher y_lat (fit target)", t_taps["y_lat"], layer_idx))
            entry["prefix_lineage"] = self.prefix_lineage(layer_idx)
            marker = self.read_done_marker(layer_idx, installed=self.installed) if args.resume else None
            if marker is not None:
                dense = self.load_ffn(marker["artefact"])
                prior_entry = self.receipt["layers"].get(str(layer_idx)) or {}
                metrics = marker.get("fold_metrics")
                source = "done_marker"
                if not metrics:
                    metrics = {k: prior_entry[k] for k in FOLD_METRIC_KEYS if k in prior_entry}
                    source = "previous_receipt" if metrics else None
                entry["fit_skipped"] = {"done_marker": self.done_marker_path(layer_idx),
                                        "artefact": marker["artefact"], "artefact_sha256": marker["artefact_sha256"],
                                        "note": "resume: the fold on disk is reused, only the streams advance"}
                entry["ffn"] = {"path": marker["artefact"], "sha256": marker["artefact_sha256"], "reused": True,
                                "bytes": os.path.getsize(marker["artefact"]),
                                "width_installed": int(dense["w1_s"].shape[0]),
                                "header_parameters": marker.get("header_parameters",
                                                                (prior_entry.get("ffn") or {}).get("header_parameters")),
                                "parameters": marker.get("parameters", (prior_entry.get("ffn") or {}).get("parameters")),
                                "tensor_sha256": marker.get("tensor_sha256")}
                entry.update(metrics)
                entry["fold_metrics_source"] = source
                if source is None:
                    entry["fold_metrics_missing"] = ("the done marker predates fold_metrics and no previous receipt "
                                                     "carries this layer; refit (drop the marker) to recover them")
                self.installed[layer_idx] = marker["artefact_sha256"]
            else:
                entry["fit_skipped"] = None
                fit_record, dense = self.fit_layer(layer_idx, weights, raw_body, cfg, t_taps, f_latent)
                entry.update(fit_record)
                entry["fold_metrics_source"] = "fit"
                self.installed[layer_idx] = fit_record["ffn"]["sha256"]
            with torch.no_grad():
                f_prefix_new, f_taps = ffn_half(runner, f_carried, f_ffn_in, cfg, device=self.device,
                                                token_chunk=token_chunk, dense=dense, latent=f_latent)
                # G3's reference: the SAME folded stream with a child that writes y_lat = 0, on the
                # held rows, this card, this dtype -- what the next layer would see from nothing
                entry["zero_child_prefix"] = self.zero_child_prefix(weights, f_carried, f_ffn_in,
                                                                    new_teacher.prefix, cfg, token_chunk)
            del f_carried, f_ffn_in, f_taps, f_latent
            new_folded = Pair(f_prefix_new.to(self.resid_device), f_blocks_after)
            entry["folded"] = {"cloned_from_teacher": False, "dense_channels": int(dense["w1_s"].shape[0])}
            if args.held_rows:
                entry["held_rows"] = self.write_held_rows(layer_idx, t_held_moe_x, f_held_moe_x, held_rows)
            del dense
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        entry["folded_seconds"] = round(time.time() - fold_start, 2)
        # both pairs leave this layer finite, or the pass stops here by name
        entry["finite_checks"].append(require_finite("teacher prefix after the layer", new_teacher.prefix, layer_idx))
        entry["finite_checks"].append(require_finite("folded prefix after the layer", new_folded.prefix, layer_idx))

        # ---- divergence of the two streams after this layer ----------------------
        with torch.no_grad():
            entry["stream_divergence"] = self.stream_divergence(new_teacher, new_folded)
        # ---- the measured acceptance rule (a fitted MoE layer installs here, or not) ----
        if weights.is_moe:
            self.accept_layer(layer_idx, entry)
        entry["prefix_out_sha256"] = {"teacher": sha256_tensor(new_teacher.prefix),
                                      "folded": sha256_tensor(new_folded.prefix)}
        entry["block_residual_blocks"] = int(new_teacher.blocks.shape[1])
        entry["peak_rss_bytes"] = self.require_peak_rss("layer_record", layer_idx)
        entry["layer_seconds"] = round(time.time() - layer_start, 1)
        entry["_teacher"] = new_teacher
        entry["_folded"] = new_folded
        entry["_folded_is_teacher"] = bool(folded_is_teacher and not weights.is_moe)
        del weights, runner, t_taps
        return entry

    # -- the measured acceptance rule ------------------------------------------------------
    def zero_child_prefix(self, weights: LayerWeights, carried: torch.Tensor, ffn_in: torch.Tensor,
                          teacher_prefix: torch.Tensor, cfg: dict, token_chunk: int) -> dict[str, Any]:
        """Folded-stream prefix after this layer if the child wrote y_lat = 0, vs the teacher, held rows.

        Same rows, same cap (``--divergence-rows``), same card and dtype as
        ``stream_divergence``'s ``held`` prefix score, so G3 compares like with like.
        """
        rows = torch.nonzero(self.corpus.held).flatten()
        piece = rows[: int(self.args.divergence_rows)] if self.args.divergence_rows > 0 else rows
        dev_rows = piece.to(self.device)
        x = ffn_in[dev_rows]
        latent_dim = int(weights.body["block_sparse_moe.routed_expert_down_proj.weight"].shape[0])
        y_zero = torch.zeros(x.shape[0], latent_dim, dtype=torch.float32, device=x.device)
        prefix_zero = carried[dev_rows] + moe_write(weights, x, y_zero, cfg, token_chunk)
        teacher_rows = teacher_prefix[piece].to(self.device).float()
        scored = write_scores(prefix_zero.float(), teacher_rows)
        scored["rows"] = int(piece.numel())
        scored["scored"] = "prefix_sum"
        del x, y_zero, prefix_zero, teacher_rows
        return {"held": scored,
                "what": "the folded stream's prefix after this layer with a child that writes y_lat = 0 "
                        "(shared experts, norms and residual verbatim), relL2 vs the teacher prefix on the "
                        "held rows; G3's reference"}

    def accept_layer(self, layer_idx: int, entry: dict[str, Any]) -> None:
        """G1-G3 on a fitted layer; writes the done marker on acceptance, removes this run's own
        ffn.safetensors and raises a named ``Refusal`` (numbers in the message and the receipt) otherwise.
        A resumed layer keeps its eligible verdict; read_done_marker checks gate mode."""
        rung = int(self.args.rung)
        if entry.get("fit_skipped") is not None:
            if not isinstance(entry.get("acceptance"), dict):
                entry["acceptance"] = {"version": ACCEPTANCE_VERSION, "verdict": "not_evaluated",
                                       "note": "done marker predates the measured acceptance rule (tool < 1.2.0); "
                                               "drop the marker to refit and gate"}
            if not isinstance(entry.get("law_verdict"), dict):
                entry["law_verdict"] = None
            entry["acceptance"]["zero_child_prefix_this_run"] = entry.get("zero_child_prefix", {}).get("held")
            entry["acceptance"]["mode_this_run"] = str(self.args.acceptance)
            entry["acceptance"]["enforced_this_run"] = bool(self.args.acceptance == "gate")
            return
        heldout = entry["fold"]["heldout"]
        acceptance = evaluate_acceptance(heldout, entry["stream_divergence"].get("held"),
                                         entry.get("zero_child_prefix", {}).get("held"),
                                         mode=str(self.args.acceptance))
        acceptance["layer"] = layer_idx
        acceptance["rung"] = rung
        entry["acceptance"] = acceptance
        ridge = entry["fold"]["ridge"]
        gates = acceptance["gates"]
        law = entry.get("law_verdict") or {}

        def fmt(value) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        log(f"L{layer_idx:02d} K={rung} acceptance {acceptance['verdict'].upper()}"
            f"{'' if acceptance['enforced'] else ' (record only, not enforced)'}: "
            f"held y_lat relL2 {fmt(_score_relL2(heldout.get('y_lat_heldout')))} "
            f"(runtime dtype {fmt(_score_relL2(heldout.get('y_lat_heldout_runtime_dtype')))}), "
            f"fit {fmt(_score_relL2(heldout.get('y_lat_fit')))}, held/fit {fmt(gates['G2_held_over_fit']['value'])} "
            f"(<= {fmt(gates['G2_held_over_fit']['threshold'])}); routed_write held "
            f"{fmt(_score_relL2(heldout.get('routed_write_heldout')))} (<= {fmt(gates['G1_routed_write_heldout']['threshold'])}); "
            f"prefix held {fmt(gates['G3_prefix_vs_zero_child']['value'])} vs zero-child "
            f"{fmt(gates['G3_prefix_vs_zero_child'].get('zero_child'))} (<= {fmt(gates['G3_prefix_vs_zero_child']['threshold'])}); "
            f"baselines zero {fmt(_score_relL2((heldout.get('baselines') or {}).get('zero', {}).get('y_lat_heldout')))} "
            f"prior {fmt(_score_relL2((heldout.get('baselines') or {}).get('prior', {}).get('y_lat_heldout')))}; "
            f"lambda rel {ridge.get('lambda_relative')} (edge {ridge.get('chosen_at_ladder_edge')}); "
            f"THE LAW would_refuse {law.get('would_refuse')} (update_scale {fmt(ridge.get('update_scale'))}, "
            f"free {fmt(ridge.get('free_solve_update_scale'))})")
        if acceptance["verdict"] == "refused" and acceptance["enforced"]:
            removed = []
            ffn_path = (entry.get("ffn") or {}).get("path")
            if ffn_path and os.path.exists(ffn_path):
                os.remove(ffn_path)  # this run's own artefact; snapshots and other runs' files are untouched
                removed.append(ffn_path)
            self.installed.pop(layer_idx, None)
            acceptance["removed"] = removed
            message = acceptance_message(layer_idx, rung, acceptance)
            log(message)
            raise Refusal(message, {
                "gate": acceptance["failing"], "version": ACCEPTANCE_VERSION,
                "gates": {name: {k: gates[name][k] for k in ("value", "threshold", "pass")} for name in acceptance["failing"]},
                "heldout": {k: heldout.get(k) for k in ("y_lat_heldout", "y_lat_heldout_runtime_dtype",
                                                         "routed_write_heldout", "y_lat_fit", "baselines")},
                "ridge": {k: ridge.get(k) for k in ("lambda_relative", "update_scale", "free_solve_update_scale",
                                                     "chosen_at_ladder_edge", "law_verdict")},
                "stream_divergence_held": entry["stream_divergence"].get("held"),
                "zero_child_prefix_held": entry.get("zero_child_prefix", {}).get("held"),
                "removed": removed, "done_marker_written": False,
            })
        # accepted (or record-only): the done marker carries the metrics INCLUDING the verdict
        marker_extra = entry["ffn"].pop("marker_extra")
        metrics = {k: entry[k] for k in FOLD_METRIC_KEYS}
        assert set(metrics) == set(FOLD_METRIC_KEYS), sorted(set(metrics) ^ set(FOLD_METRIC_KEYS))
        marker_extra["fold_metrics"] = metrics
        entry["ffn"]["done_marker_sha256"] = self.write_done_marker(
            layer_idx, entry["ffn"]["path"], entry["ffn"]["sha256"], marker_extra)

    def window_acceptance(self, eos: dict[str, Any]) -> dict[str, Any]:
        """G4: end-of-stack held top-1 / KL per fitted layer vs the first measured window's calibration.

        The first window with an end_of_stack number writes ``acceptance_calibration.json``
        beside the receipts (context, rung and installed prefix stamped); later windows in
        that same context may extend the unchanged calibrated prefix. The constants may not
        be raised silently: they live in
        ``ACCEPTANCE_CONSTANTS`` and every receipt names them.
        """
        held = eos.get("held") or {}
        # Snapshots can carry a validated prefix even when its earlier receipt is
        # absent. The installed state, not receipt coverage, defines the child.
        folded = sorted(self.installed)
        n_fitted = max(len(folded), 1)
        kl = held.get("kl_parent_child_mean")
        top1 = held.get("top1_agreement")
        budget = float(ACCEPTANCE_CONSTANTS["G4_window_kl_per_layer_budget_x"])
        floor = float(ACCEPTANCE_CONSTANTS["G4_window_top1_floor_x"])
        record: dict[str, Any] = {
            "version": ACCEPTANCE_VERSION, "gate": "G4_window_intelligence", "layers_folded": folded,
            "held_kl_parent_child_mean": kl, "held_top1_agreement": top1,
            "raw_held_kl_parent_child_mean": kl, "kl_roundoff_policy": dict(KL_ROUNDOFF_POLICY),
            "receipt_layers_missing": [k for k in folded if str(k) not in self.receipt["layers"]],
            "kl_per_fitted_layer": None,
            "constants": {"G4_window_kl_per_layer_budget_x": budget, "G4_window_top1_floor_x": floor},
            "mode": str(self.args.acceptance), "enforced": bool(self.args.acceptance == "gate"),
            "gates": {},
        }
        def unavailable(note: str) -> dict[str, Any]:
            record.update({"verdict": "refused" if record["enforced"] else "not_evaluated", "note": note})
            if record["enforced"]:
                record["message"] = "window refused by G4: " + note
            return record

        def valid_score(value: Any, *, probability: bool = False) -> bool:
            return (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0 and (not probability or value <= 1))

        if (isinstance(kl, (int, float)) and not isinstance(kl, bool) and math.isfinite(kl)
                and -KL_ROUNDOFF_POLICY["negative_absolute_tolerance"] <= kl < 0):
            kl = 0.0
            record["held_kl_parent_child_mean"] = kl
            record["kl_roundoff_normalized"] = True
        if not valid_score(kl) or not valid_score(top1, probability=True):
            return unavailable("end_of_stack carries unavailable or invalid held numbers")
        if not folded:
            return unavailable("no installed folded layers to calibrate or evaluate")
        record["kl_per_fitted_layer"] = float(kl) / n_fitted
        path = os.path.join(self.receipt_dir, ACCEPTANCE_CALIBRATION_NAME)
        if os.path.exists(path):
            record["calibration_path"] = path
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    calibration = json.load(handle)
            except (OSError, ValueError):
                return unavailable("incompatible unreadable calibration; not overwritten")
            if not isinstance(calibration, dict):
                return unavailable("incompatible non-object calibration; not overwritten")
            baseline = calibration.get("installed_ffn_sha256")
            baseline_layers = calibration.get("layers_folded")
            current_installed = {str(k): v for k, v in sorted(self.installed.items())}
            if (calibration.get("version") != ACCEPTANCE_VERSION
                    or calibration.get("constants") != ACCEPTANCE_CONSTANTS
                    or calibration.get("kl_roundoff_policy") != KL_ROUNDOFF_POLICY
                    or calibration.get("resume_context") != self.resume_context
                    or calibration.get("corpus_sha256") != self.corpus.sha256
                    or not isinstance(calibration.get("rung"), int) or isinstance(calibration["rung"], bool)
                    or calibration["rung"] != int(self.args.rung)
                    or not isinstance(baseline, dict) or not baseline
                    or not isinstance(baseline_layers, list)
                    or any(not isinstance(k, int) or isinstance(k, bool) or k < 0 for k in baseline_layers)
                    or len(set(baseline_layers)) != len(baseline_layers)
                    or set(baseline_layers) != self.prior_moe_layers(max(baseline_layers, default=-1) + 1)
                    or set(baseline) != {str(k) for k in baseline_layers}
                    or any(not isinstance(v, str) or len(v) != 64
                           or any(c not in "0123456789abcdef" for c in v) for v in baseline.values())
                    or any(current_installed.get(k) != v for k, v in baseline.items())
                    or not valid_score(calibration.get("kl_per_fitted_layer"))
                    or not valid_score(calibration.get("held_top1_agreement"), probability=True)):
                return unavailable("incompatible calibration context, version, constants, prefix or metrics; "
                                   "not gated against stale numbers and not overwritten")
            thr_kl = budget * float(calibration["kl_per_fitted_layer"])
            thr_top1 = floor * float(calibration["held_top1_agreement"])
            if not math.isfinite(thr_kl) or not math.isfinite(thr_top1):
                return unavailable("incompatible calibration produces non-finite thresholds; not overwritten")
            gates = {
                "kl_per_fitted_layer": {"value": record["kl_per_fitted_layer"], "threshold": thr_kl,
                                        "pass": bool(record["kl_per_fitted_layer"] <= thr_kl)},
                "held_top1_agreement": {"value": float(top1), "threshold": thr_top1,
                                        "pass": bool(float(top1) >= thr_top1)},
            }
            failing = [k for k, g in gates.items() if not g["pass"]]
            record.update({"calibration": calibration, "calibration_path": path, "gates": gates,
                           "verdict": "accepted" if not failing else "refused"})
            if failing:
                record["message"] = ("window refused by G4 (" + ACCEPTANCE_VERSION + "): " + "; ".join(
                    f"{k} {gates[k]['value']:.6g} vs threshold {gates[k]['threshold']:.6g}" for k in failing)
                    + f" (calibration {path})")
            return record
        calibration = {
            "version": ACCEPTANCE_VERSION, "corpus_sha256": self.corpus.sha256, "rung": int(self.args.rung),
            "constants": dict(ACCEPTANCE_CONSTANTS), "resume_context": self.resume_context,
            "kl_roundoff_policy": dict(KL_ROUNDOFF_POLICY),
            "raw_held_kl_parent_child_mean": record["raw_held_kl_parent_child_mean"],
            "installed_ffn_sha256": {str(k): self.installed[k] for k in folded},
            "layers_folded": folded, "kl_per_fitted_layer": record["kl_per_fitted_layer"],
            "held_kl_parent_child_mean": float(kl), "held_top1_agreement": float(top1),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool_version": TOOL_VERSION,
            "note": "first measured window: later windows at this corpus / rung gate against these numbers "
                    "(KL per fitted layer <= budget x, top-1 >= floor x); may not be raised silently",
        }
        write_receipt(path, calibration)
        record.update({"verdict": "calibrated", "calibration_written": path, "calibration": calibration})
        return record

    # -- layer 0 verbatim ------------------------------------------------------------
    def write_dense_layer_verbatim(self, layer_idx: int, weights: LayerWeights,
                                   raw_body: dict[str, torch.Tensor]) -> dict[str, Any]:
        """`dense/L{l}/mlp.safetensors`: the dense KimiMLP copied byte-for-byte (test 5)."""
        names = [k for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight") if k in raw_body]
        if not names:
            raise SystemExit(f"layer {layer_idx} is not a MoE layer and carries no mlp.* tensors")
        prefix = self.index.layer_prefix_for(layer_idx)
        full_names = [f"{prefix}{n}" for n in names]
        shard_paths = {}
        for name in full_names:
            shard = self.index.weight_map[name]
            shard_paths.setdefault(shard, []).append(name)
        raw_records: dict[str, dict[str, Any]] = {}
        for shard, wanted in shard_paths.items():
            raw_records.update(read_raw_tensor_records(self.stager.path_for(shard), wanted))
        listing = self.layer_tensor_listing(layer_idx)
        out_path = os.path.join(self.layer_dir(layer_idx), "mlp.safetensors")
        payload = {n: raw_body[n].cpu().contiguous() for n in names}
        digest = save_tensors(out_path, payload, metadata={
            "layer": str(layer_idx), "kind": "dense_mlp_verbatim", "tool": TOOL_NAME,
            "input_ids_sha256": self.corpus.sha256})
        # what was written must hash exactly as what was read
        written = read_raw_tensor_records(out_path, names)
        per_tensor = {}
        for short, full in zip(names, full_names):
            src = raw_records[full]
            dst = written[short]
            per_tensor[full] = {"dtype": src["dtype"], "shape": src["shape"], "elements": int(payload[short].numel()),
                                "sha256_import_lane": src["sha256_import_lane"],
                                "written_sha256_import_lane": dst["sha256_import_lane"],
                                "byte_exact": src["sha256_import_lane"] == dst["sha256_import_lane"]}
        if not all(v["byte_exact"] for v in per_tensor.values()):
            raise SystemExit(f"layer {layer_idx} verbatim copy is not byte-exact: {per_tensor}")
        comparison = self.compare_layer0_receipt(layer_idx, per_tensor)
        record = {"path": out_path, "sha256": digest, "tensors": per_tensor,
                  "header_parameters": int(sum(t.numel() for t in payload.values())),
                  "import_receipt_comparison": comparison, "tensor_listing": listing,
                  "tensor_sha256": {full: per_tensor[full]["sha256_import_lane"] for full in full_names}}
        marker_sha = self.write_done_marker(layer_idx, out_path, digest, {"moe": False, "kind": "dense_mlp_verbatim"})
        record["done_marker_sha256"] = marker_sha
        return {"mlp": record}

    # -- --body-dir: the non-expert body, verbatim, for the OC03 assembly --------------
    def _verbatim_file(self, out_path: str, full_names: list[str], payload: dict[str, torch.Tensor],
                       metadata: dict[str, str], from_source: bool = False) -> dict[str, Any]:
        """Write ``payload`` (short name -> host tensor) and prove every tensor byte-exact
        against its shard with the import lane's sha (dtype + shape + bytes).  Layer shards
        go through the stager (they are in the plan); model-level shards are read from the
        source (``from_source``) so they never take a window slot."""
        shard_paths: dict[str, list[str]] = {}
        for name in full_names:
            shard_paths.setdefault(self.index.weight_map[name], []).append(name)
        raw_records: dict[str, dict[str, Any]] = {}
        for shard, wanted in shard_paths.items():
            path = os.path.join(self.args.src_dir, shard) if from_source else self.stager.path_for(shard)
            raw_records.update(read_raw_tensor_records(path, wanted))
        digest = save_tensors(out_path, payload, metadata=metadata)
        written = read_raw_tensor_records(out_path, list(payload))
        tensor_sha: dict[str, str] = {}
        mismatched = []
        for short, full in zip(payload, full_names):
            ours = written[short]["sha256_import_lane"]
            tensor_sha[full] = ours
            if ours != raw_records[full]["sha256_import_lane"]:
                mismatched.append(full)
        if mismatched:
            raise SystemExit(f"{out_path}: verbatim copy is not byte-exact for {mismatched}")
        return {"path": out_path, "sha256": digest, "bytes": os.path.getsize(out_path), "tensors": len(payload),
                "header_parameters": int(sum(t.numel() for t in payload.values())),
                "byte_exact": True, "tensor_sha256": tensor_sha}

    def write_body_verbatim(self, layer_idx: int, raw_body: dict[str, torch.Tensor]) -> dict[str, Any]:
        """`<body-dir>/L{l}/body.safetensors`: attention + norms + residual maps of the layer,
        byte-for-byte; never an expert, the router, the MoE block or the dense MLP."""
        names = sorted(k for k in raw_body if not k.startswith(BODY_EXCLUDED_PREFIXES))
        if not any(n.startswith("self_attn.") for n in names):
            raise SystemExit(f"layer {layer_idx}: the raw read carried no self_attn.* tensors for --body-dir")
        prefix = self.index.layer_prefix_for(layer_idx)
        out_path = os.path.join(os.path.abspath(self.args.body_dir), f"L{layer_idx:02d}", "body.safetensors")
        payload = {n: raw_body[n].cpu().contiguous() for n in names}
        record = self._verbatim_file(out_path, [f"{prefix}{n}" for n in names], payload, metadata={
            "layer": str(layer_idx), "kind": "non_expert_body_verbatim", "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION, "excluded": ".experts.*, " + ", ".join(BODY_EXCLUDED_PREFIXES) + "*"})
        record["excluded_prefixes"] = list(BODY_EXCLUDED_PREFIXES) + [".experts."]
        return record

    def write_model_level_verbatim(self) -> dict[str, Any]:
        """`<body-dir>/model_level.safetensors`: embedding, final norm, output attn-res read, lm_head."""
        names = sorted(self.index.model_level_keys())
        if not names:
            raise SystemExit("the index carries no model-level tensors for --body-dir")
        out_path = os.path.join(os.path.abspath(self.args.body_dir), "model_level.safetensors")
        payload = {}
        for name in names:
            shard = self.index.weight_map[name]
            payload[name] = read_tensors(os.path.join(self.args.src_dir, shard), [name])[name].cpu().contiguous()
            self.stager.note_source_read(shard, name)
        record = self._verbatim_file(out_path, names, payload, metadata={
            "kind": "model_level_verbatim", "tool": TOOL_NAME, "tool_version": TOOL_VERSION}, from_source=True)
        record["names"] = names
        return record

    def compare_layer0_receipt(self, layer_idx: int, per_tensor: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Run-book test 5 on the real invocation: the receipt defaults to the import lane's
        k3_layer0_import_receipt.json and its absence is a refusal unless --no-layer0-receipt
        says so out loud (synthetic fixtures, a parent with no import receipt)."""
        path = self.args.layer0_import_receipt
        if self.args.no_layer0_receipt:
            return {"compared": False, "path": path, "skipped_by_flag": "--no-layer0-receipt",
                    "reason": "--no-layer0-receipt was given explicitly; the verbatim copy is byte-checked "
                              "against the shard only"}
        if not path:
            raise SystemExit("--layer0-import-receipt is empty; name the k3_layer0_import_receipt.json to sha-match "
                             "the verbatim layer-0 MLP against, or pass --no-layer0-receipt explicitly")
        if not os.path.exists(path):
            raise SystemExit(f"--layer0-import-receipt {path} does not exist; run-book test 5 (layer-0 verbatim "
                             "sha-match) is not skipped silently -- point the flag at the receipt or pass "
                             "--no-layer0-receipt explicitly")
        with open(path, "r", encoding="utf-8") as handle:
            got = json.load(handle)
        consumed: list[dict[str, Any]] = []
        if "columns" in got:
            for column in got["columns"]:
                consumed = column.get("import_receipt", {}).get("consumed", [])
                if consumed:
                    break
        elif "consumed" in got:
            consumed = got["consumed"]
        by_name = {c["name"]: c for c in consumed}
        if int(got.get("layer_index", layer_idx)) != layer_idx:
            return {"compared": False, "path": path,
                    "reason": f"the receipt is for layer {got.get('layer_index')}, this is layer {layer_idx}"}
        matches = {}
        for name, rec in per_tensor.items():
            ref = by_name.get(name)
            matches[name] = {"in_receipt": ref is not None,
                             "receipt_sha256": ref["sha256"] if ref else None,
                             "ours": rec["sha256_import_lane"],
                             "match": bool(ref and ref["sha256"] == rec["sha256_import_lane"])}
        all_match = all(m["match"] for m in matches.values())
        if not all_match:
            raise SystemExit(f"layer {layer_idx} verbatim tensors do not sha-match {path}: "
                             f"{json.dumps(matches, indent=1)}")
        return {"compared": True, "path": path, "receipt_sha256": sha256_file(path),
                "tensors_compared": len(matches), "all_match": True, "matches": matches}

    def layer_tensor_listing(self, layer_idx: int) -> dict[str, Any]:
        """`importing.kimi_k3.k3_layer_tensor_names` cross-checked against the index, if importable."""
        index_names = sorted(self.index.layer_keys(layer_idx))
        record: dict[str, Any] = {"index_tensors": len(index_names), "chimera_src": self.args.chimera_src}
        src = self.args.chimera_src
        if not src or not os.path.isdir(src):
            record["chimera_listing"] = "not attempted (no --chimera-src directory)"
            return record
        try:
            if src not in sys.path:
                sys.path.insert(0, src)
            from chimera_omni.importing.kimi_k3 import k3_layer_tensor_names  # type: ignore
            from chimera_omni.importing.profiles import k3_chimera_config  # type: ignore

            sizes = [int(self.cfg.get("intermediate_size", 0))] * self.index.num_layers
            chimera_cfg = k3_chimera_config(self.index.config_json, dense_intermediate_sizes=sizes)
            names = k3_layer_tensor_names(chimera_cfg, layer_idx, prefix=self.index.model_prefix,
                                          dense_mlp=not self.index.is_moe_layer(layer_idx))
            missing = [n for n in names if n not in self.index.weight_map]
            record.update({"chimera_listing": names, "chimera_listing_count": len(names),
                           "missing_from_index": missing, "all_in_index": not missing})
        except Exception as exc:  # recorded, never fatal: the listing is a cross-check
            record["chimera_listing_error"] = repr(exc)[:300]
        return record

    # -- the fit -----------------------------------------------------------------------
    def fit_layer(self, layer_idx: int, weights: LayerWeights, raw_body: dict[str, torch.Tensor], cfg: dict,
                  t_taps: dict[str, Any], f_latent: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        args = self.args
        cache = {"latent_in": f_latent, "y_lat": t_taps["y_lat"],
                 "router_topk_idx": t_taps["router_topk_idx"], "router_topk_w": t_taps["router_topk_w"]}
        max_device_gb = (c14.default_device_budget_gb(self.device) if args.max_device_gb < 0
                         else float(args.max_device_gb))
        fold = DualCoresetFold(weights, cache, cfg, self.device, self.corpus.fit, self.corpus.held,
                               check_blocks=int(args.holdout_blocks), chunk=int(args.chunk),
                               max_gram_gb=float(args.max_gram_gb), max_device_gb=max_device_gb)
        layer_dir = self.layer_dir(layer_idx)
        os.makedirs(layer_dir, exist_ok=True)
        rung = int(args.rung)
        try:
            rung_receipt = c14.run_rung(fold, rung, int(args.k_pool_factor), layer_dir, layer_idx, cfg,
                                        self.index.hidden_size, form_m=False)
        except torch.linalg.LinAlgError as exc:  # a Gram defect is named, never swallowed
            raise SystemExit(f"layer {layer_idx} K={rung}: the fp64 Gram refit failed in the factorisation "
                             f"({exc}); this is a Gram defect (section 3b), not a fit failure to paper over") from exc
        rung_receipt["prefix_lineage"] = self.prefix_lineage(layer_idx)
        form_e_path = rung_receipt["form_E_path"]
        from safetensors.torch import load_file

        form_e = load_file(form_e_path)
        dense = {"w1_s": form_e["w1_s"], "w3_s": form_e["w3_s"], "w2p": form_e["w2p"]}
        # the fold metrics the S2 gate reads; persisted in the done marker so a resume
        # over the marker carries them (they were lost on reuse before)
        metrics: dict[str, Any] = {
            "fold": {k: v for k, v in rung_receipt.items() if k not in ("form_E_path",)},
            "pivot_seconds": round(fold.timings.get("stage1_seconds", 0.0) + fold.timings.get("stage2_seconds", 0.0), 2),
            "fit_timings": dict(fold.timings),
            "veto_binding": bool(rung_receipt["ridge"]["veto_binding"]),
            "update_scale": float(rung_receipt["ridge"]["update_scale"]),
            "fit_relL2_heldout": float(rung_receipt["heldout"]["y_lat_heldout"]["relL2"]),
            "routed_write_relL2_heldout": float(rung_receipt["heldout"]["routed_write_heldout"]["relL2"]),
            "width_installed": int(rung_receipt["width_installed"]),
            "law_verdict": rung_receipt["ridge"].get("law_verdict"),
        }
        # `acceptance` joins in accept_layer (it needs the folded prefix); the done marker is written there
        assert set(metrics) | {"acceptance"} == set(FOLD_METRIC_KEYS), sorted(set(metrics) ^ set(FOLD_METRIC_KEYS))
        ffn_record = self.write_ffn(layer_idx, form_e, raw_body, rung_receipt, cfg)
        os.remove(form_e_path)  # this run's own intermediate; the same bytes live inside ffn.safetensors
        record: dict[str, Any] = {
            **metrics,
            "fold_intermediate_removed": {"path": form_e_path, "sha256": rung_receipt["form_E_sha256"]},
            "ffn": ffn_record,
        }
        if args.arm_rung and layer_idx in parse_layers(args.arm_layers, self.index.num_layers):
            arm_dir = os.path.join(self.out, f"arm_{int(args.arm_rung)}", f"L{layer_idx:02d}")
            os.makedirs(arm_dir, exist_ok=True)
            arm = c14.run_rung(fold, int(args.arm_rung), int(args.k_pool_factor), arm_dir, layer_idx, cfg,
                               self.index.hidden_size, form_m=False)
            record["arm"] = {"rung": int(args.arm_rung), "path": arm["form_E_path"], "sha256": arm["form_E_sha256"],
                             "width_installed": arm["width_installed"], "ridge": arm["ridge"],
                             "heldout": arm["heldout"], "note": "diagnostic arm; the folded stream carries the main rung"}
        if args.chimera_src and layer_idx == self.receipt.get("first_moe_layer"):
            record["dequant_crosscheck"] = self.dequant_crosscheck(weights)
        if args.mixer_diagnostic and layer_idx == self.receipt.get("first_moe_layer"):
            record["mixer_diagnostic"] = self.mixer_diagnostic(weights, f_latent.shape[0])
        del fold, form_e
        return record, dense

    def write_ffn(self, layer_idx: int, form_e: dict[str, torch.Tensor], raw_body: dict[str, torch.Tensor],
                  rung_receipt: dict[str, Any], cfg: dict) -> dict[str, Any]:
        """`dense/L{l}/ffn.safetensors`: form E + the verbatim latent maps, norm and shared experts.

        The done marker is NOT written here any more (v1.2.0): ``accept_layer`` writes it once
        the measured acceptance rule passes, so a marker means "installed AND accepted".
        """
        verbatim = ("block_sparse_moe.routed_expert_down_proj.weight",
                    "block_sparse_moe.routed_expert_up_proj.weight",
                    "block_sparse_moe.routed_expert_norm.weight",
                    "block_sparse_moe.shared_experts.gate_proj.weight",
                    "block_sparse_moe.shared_experts.up_proj.weight",
                    "block_sparse_moe.shared_experts.down_proj.weight")
        missing = [n for n in verbatim if n not in raw_body]
        if missing:
            raise SystemExit(f"layer {layer_idx}: the shard read carried no {missing}")
        payload = {"w1_s": form_e["w1_s"].contiguous(), "w3_s": form_e["w3_s"].contiguous(),
                   "w2p": form_e["w2p"].contiguous(), "channels": form_e["channels"].contiguous()}
        for name in verbatim:
            payload[name] = raw_body[name].cpu().contiguous()
        width = int(payload["w1_s"].shape[0])
        latent_dim = int(payload["w1_s"].shape[1])
        accounting = c14.parameter_accounting(cfg, width, latent_dim, self.index.hidden_size)
        header_params = int(sum(t.numel() for k, t in payload.items() if k != "channels"))
        if header_params != accounting["child_moe_total"]:
            raise SystemExit(f"layer {layer_idx}: ffn.safetensors header count {header_params} != "
                             f"child_moe_total {accounting['child_moe_total']} from parameter_accounting")
        path = os.path.join(self.layer_dir(layer_idx), "ffn.safetensors")
        digest = save_tensors(path, payload, metadata={
            "form": "E", "layer": str(layer_idx), "K": str(int(self.args.rung)), "width_installed": str(width),
            "latent_dim": str(latent_dim), "hidden": str(self.index.hidden_size),
            "shared_channels": str(int(payload["block_sparse_moe.shared_experts.gate_proj.weight"].shape[0])),
            "child_moe_total": str(accounting["child_moe_total"]),
            "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "input_ids_sha256": self.corpus.sha256,
            "routed_expert_norm_zero_gains": str(int((payload["block_sparse_moe.routed_expert_norm.weight"].float() == 0).sum())),
            "form_E_stage_sha256": rung_receipt["form_E_sha256"],
        })
        tensor_sha = {k: sha256_tensor(v) for k, v in payload.items() if v.is_floating_point()}
        tensor_sha["channels"] = sha256_bytes(payload["channels"].contiguous().numpy().tobytes())
        marker_extra = {
            "moe": True, "kind": "form_E_ffn", "width_installed": width,
            "child_moe_total": accounting["child_moe_total"], "tensor_sha256": tensor_sha,
            "header_parameters": header_params, "bytes": os.path.getsize(path), "parameters": accounting}
        return {"path": path, "sha256": digest, "bytes": os.path.getsize(path), "width_installed": width,
                "header_parameters": header_params, "bookkeeping_elements": int(payload["channels"].numel()),
                "parameters": accounting, "done_marker_sha256": None, "tensor_sha256": tensor_sha,
                "marker_extra": marker_extra,
                "done_marker_note": "written by accept_layer after the measured acceptance rule; a refused "
                                    "layer leaves no marker"}

    def load_ffn(self, path: str) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        got = load_file(path)
        return {"w1_s": got["w1_s"].to(self.device), "w3_s": got["w3_s"].to(self.device),
                "w2p": got["w2p"].to(self.device)}

    def write_held_rows(self, layer_idx: int, teacher_rows: torch.Tensor | None, folded_rows: torch.Tensor | None,
                        held_positions: torch.Tensor) -> dict[str, Any]:
        path = os.path.join(self.out, "resid", f"L{layer_idx:02d}", "held_rows.safetensors")
        payload = {"held_positions": held_positions.to(torch.int64)}
        if teacher_rows is not None:
            payload["teacher_moe_x"] = teacher_rows.contiguous()
        if folded_rows is not None:
            payload["folded_moe_x"] = folded_rows.contiguous()
        digest = save_tensors(path, payload, metadata={"layer": str(layer_idx), "tool": TOOL_NAME,
                                                       "input_ids_sha256": self.corpus.sha256,
                                                       "what": "FFN-input rows (post-attention residual) of the held tokens"})
        return {"path": path, "sha256": digest, "rows": int(held_positions.numel()),
                "bytes": os.path.getsize(path)}

    def stream_divergence(self, teacher: Pair, folded: Pair) -> dict[str, Any]:
        """Folded-vs-teacher after the layer, per mask: the prefix_sum AND the block stack.

        On snapshot layers (``l % block_size == 0``) the prefix is reset to attn_out
        and the diverged prefix is pushed into the block stack, so the prefix score
        alone would read near-zero there; ``block_stack`` scores the stacked residuals
        (all blocks, flattened) so the cumulative divergence stays visible on every
        layer.  Diagnostic only (K3 rules 4/5): nothing here gates.
        """
        out: dict[str, Any] = {"what": "prefix_sum = the carried residual; block_stack = every block residual "
                                       "flattened (the AttnRes read's other input); relL2 of folded vs teacher"}
        n_blocks = int(teacher.blocks.shape[1])
        for name, mask in (("held", self.corpus.held), ("fit", self.corpus.fit), ("real", self.corpus.real)):
            rows = torch.nonzero(mask).flatten()
            if rows.numel() == 0:
                out[name] = None
                continue
            piece = rows[: int(self.args.divergence_rows)] if self.args.divergence_rows > 0 else rows
            t = teacher.prefix[piece].to(self.device).float()
            f = folded.prefix[piece].to(self.device).float()
            out[name] = write_scores(f, t)
            out[name]["rows"] = int(piece.numel())
            out[name]["scored"] = "prefix_sum"
            del t, f
            if n_blocks > 0:
                tb = teacher.blocks[piece].to(self.device).float().reshape(int(piece.numel()), -1)
                fb = folded.blocks[piece].to(self.device).float().reshape(int(piece.numel()), -1)
                out[name]["block_stack"] = {**write_scores(fb, tb), "blocks": n_blocks}
                del tb, fb
            else:
                out[name]["block_stack"] = None
        return out

    # -- optional diagnostics ---------------------------------------------------------
    def dequant_crosscheck(self, weights: LayerWeights) -> dict[str, Any]:
        """`dequant_mxfp4` from the import lane vs ours on ONE expert tensor (bit-exact or not)."""
        try:
            src = self.args.chimera_src
            if src not in sys.path:
                sys.path.insert(0, src)
            from chimera_omni.importing.kimi_k3 import dequant_mxfp4 as lane_dequant  # type: ignore
            from k3_common import dequant_mxfp4

            expert = sorted(weights.experts)[0]
            packed = weights.experts[expert]["w1.weight_packed"].cpu()
            scale = weights.experts[expert]["w1.weight_scale"].cpu()
            ours = dequant_mxfp4(packed, scale, dtype=torch.bfloat16)
            theirs = lane_dequant(packed, scale)
            theirs = theirs.to(torch.bfloat16) if theirs.dtype != torch.bfloat16 else theirs
            return {"expert": expert, "tensor": "w1", "shape": list(ours.shape),
                    "bit_exact": bool(torch.equal(ours, theirs)),
                    "max_abs_difference": float((ours.float() - theirs.float()).abs().max()),
                    "ours_sha256": sha256_tensor(ours), "lane_sha256": sha256_tensor(theirs)}
        except Exception as exc:
            return {"error": repr(exc)[:300]}

    def mixer_diagnostic(self, weights: LayerWeights, tokens: int) -> dict[str, Any]:
        """Layer-1 KDA/MLA mixer vs the repo module (k3_receipt.check_halves, attention only)."""
        try:
            import k3_receipt

            modeling, config_module = k3_receipt.import_k3_modeling(self.args.modeling_src, self.args.src_dir)
            n = min(256, int(self.corpus.segments[0].T))
            generator = torch.Generator(device="cpu").manual_seed(0)
            hidden = (torch.randn(1, n, self.index.hidden_size, generator=generator) * 0.02)
            dtype = self.compute_dtype or torch.bfloat16
            hidden = hidden.to(dtype).to(self.device)
            ns = argparse.Namespace(attn_implementation="eager", kda_backend=self.args.kda_backend,
                                    reference_moe=False)
            halves = k3_receipt.check_halves(weights, self.cfg, hidden, modeling, config_module, ns)
            return {"attention": halves.get("attention"), "attention_error": halves.get("attention_error"),
                    "transformers_compat": dict(k3_receipt.TRANSFORMERS_COMPAT), "tokens": n}
        except Exception as exc:
            return {"error": repr(exc)[:400]}

    # -- end of stack ----------------------------------------------------------------------
    def end_of_stack(self, teacher: Pair, folded: Pair) -> dict[str, Any]:
        """top-1 / KL of folded vs teacher on the held tokens; teacher-forced logits on the review rows."""
        record: dict[str, Any] = {"held": {}, "fit": {}, "review": {}}
        agg: dict[str, dict[str, list[float]]] = {"held": {}, "fit": {}}
        review_payload: dict[str, dict[str, list[torch.Tensor]]] = {}
        review_rows: list[dict[str, Any]] = []
        topk = int(self.args.review_topk)
        with torch.no_grad():
            for seg, r0, r1, start, stop in self.corpus.row_chunks(int(self.args.row_chunk)):
                n_rows = r1 - r0
                t_logits = self.logits_rows(teacher, start, stop, n_rows, seg.T)
                f_logits = self.logits_rows(folded, start, stop, n_rows, seg.T)
                ids = seg.ids[r0:r1].to(self.device)
                if seg.role == "evidence":
                    for name, mask in (("held", seg.held_mask[r0:r1]), ("fit", seg.fit_mask[r0:r1])):
                        mask = mask.to(self.device)
                        # position t predicts ids[t+1]; both must be real tokens of the same split
                        predict = mask[:, :-1] & mask[:, 1:]
                        if not bool(predict.any()):
                            continue
                        t_sel = t_logits[:, :-1][predict]
                        f_sel = f_logits[:, :-1][predict]
                        target = ids[:, 1:][predict]
                        stats = kl_top_agreement(f_sel, t_sel)
                        stats["teacher_nll"] = float(F.cross_entropy(t_sel, target))
                        stats["folded_nll"] = float(F.cross_entropy(f_sel, target))
                        stats["positions"] = int(predict.sum())
                        for k, v in stats.items():
                            agg[name].setdefault(k, []).append(float(v))
                        del t_sel, f_sel
                else:
                    bucket = review_payload.setdefault(seg.name, {})
                    for r in range(n_rows):
                        length = int(seg.real_lengths[r0 + r])
                        if length < 2:
                            continue
                        t_row = t_logits[r, :length - 1]
                        f_row = f_logits[r, :length - 1]
                        target = ids[r, 1:length]
                        t_lp = torch.log_softmax(t_row, dim=-1)
                        f_lp = torch.log_softmax(f_row, dim=-1)
                        stats = kl_top_agreement(f_row, t_row)
                        t_nll = -t_lp.gather(1, target.unsqueeze(1)).squeeze(1)
                        f_nll = -f_lp.gather(1, target.unsqueeze(1)).squeeze(1)
                        t_top = t_lp.topk(min(topk, t_lp.shape[-1]), dim=-1)
                        f_top = f_lp.topk(min(topk, f_lp.shape[-1]), dim=-1)
                        kl_pos = (t_lp.exp() * (t_lp - f_lp)).sum(-1)
                        row_index = r0 + r
                        for key, value in (("teacher_next_token_logprob", -t_nll), ("folded_next_token_logprob", -f_nll),
                                           ("teacher_argmax", t_row.argmax(-1).to(torch.int32)),
                                           ("folded_argmax", f_row.argmax(-1).to(torch.int32)),
                                           ("teacher_topk_ids", t_top.indices.to(torch.int32)),
                                           ("teacher_topk_logprobs", t_top.values),
                                           ("folded_topk_ids", f_top.indices.to(torch.int32)),
                                           ("folded_topk_logprobs", f_top.values),
                                           ("kl_teacher_folded", kl_pos),
                                           ("row_index", torch.full((length - 1,), row_index, dtype=torch.int32)),
                                           ("position", torch.arange(length - 1, dtype=torch.int32))):
                            bucket.setdefault(key, []).append(value.cpu())
                        review_rows.append({"file": seg.name, "row": row_index, "real_tokens": length,
                                            "teacher_nll_mean": float(t_nll.mean()),
                                            "folded_nll_mean": float(f_nll.mean()), **stats})
                        del t_row, f_row, t_lp, f_lp
                del t_logits, f_logits, ids
        for name in ("held", "fit"):
            if not agg[name]:
                record[name] = None
                continue
            weights = torch.tensor(agg[name]["positions"], dtype=torch.float64)
            total = float(weights.sum())
            record[name] = {"positions": int(total)}
            for key, values in agg[name].items():
                if key == "positions":
                    continue
                vals = torch.tensor(values, dtype=torch.float64)
                if key.endswith("_max"):
                    record[name][key] = float(vals.max())
                else:
                    record[name][key] = float((vals * weights).sum() / max(total, 1.0))
        files = {}
        for name, bucket in review_payload.items():
            path = os.path.join(self.out, "review", f"{name}_teacher_forced.safetensors")
            payload = {k: torch.cat(v, dim=0).contiguous() for k, v in bucket.items()}
            digest = save_tensors(path, payload, metadata={
                "tool": TOOL_NAME, "input_ids_sha256": self.corpus.sha256, "file": name,
                "topk": str(topk), "what": ("teacher-forced next-token logprobs, argmax, top-k and per-position "
                                            "KL(teacher||folded) on the masked review rows; the teacher columns "
                                            "are the K3 R-proxy baseline")})
            files[name] = {"path": path, "sha256": digest, "positions": int(payload["position"].numel())}
        record["review"] = {"files": files, "rows": review_rows,
                            "note": "full-vocabulary logits are reduced per position; nothing here is a free-run decode"}
        return record

    # -- determinism -------------------------------------------------------------------------
    def determinism_sha256(self) -> str:
        """sha256 over the timing-free, path-free fields: what a resumed pass must reproduce."""
        core: dict[str, Any] = {"corpus_sha256": self.corpus.sha256, "rung": int(self.args.rung),
                                "layers": {}, "final_prefix_sha256": self.receipt.get("final_prefix_sha256"),
                                "end_of_stack": None}
        for key, entry in sorted(self.receipt["layers"].items(), key=lambda kv: int(kv[0])):
            artefact = entry.get("ffn") or entry.get("mlp") or {}
            core["layers"][key] = {
                "prefix_out_sha256": entry.get("prefix_out_sha256"),
                # tensor CONTENT digests, never the file's: safetensors serialises the
                # metadata map in hash order, so two byte-identical builds differ as files
                "artefact_tensor_sha256": artefact.get("tensor_sha256"),
                "width_installed": artefact.get("width_installed"),
            }
        eos = self.receipt.get("end_of_stack")
        if eos:
            core["end_of_stack"] = {name: eos.get(name) for name in ("held", "fit")}
            core["review_rows"] = [{k: v for k, v in row.items()} for row in eos.get("review", {}).get("rows", [])]
        return sha256_text(canonical_json(core))


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=DEFAULT_SRC_DIR,
                    help="where the shards are READ (index + config + weights); the Lexar copy by default")
    ap.add_argument("--model-dir", default=None, help="the recorded identity of the parent (defaults to --src-dir)")
    ap.add_argument("--input-ids", required=True, help="k3_input_ids.safetensors (in-band fit/held masks)")
    ap.add_argument("--review-input-ids", action="append", default=[],
                    help="masked review rows (k3_review_input_ids.safetensors); repeatable")
    ap.add_argument("--layers", default="0-92")
    ap.add_argument("--rung", type=int, default=17408)
    ap.add_argument("--k-pool-factor", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--experts-device", default="cpu")
    ap.add_argument("--resid-device", default="cpu", help="where the two residual pairs live between layers")
    ap.add_argument("--kda-backend", default="torch", choices=("torch", "fla"))
    ap.add_argument("--compute-dtype", default="auto", choices=("auto", "bf16", "fp32"),
                    help="auto = the parent's dtype (bf16); fp32 for exactness tests and CPU runs")
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--stage-window", type=int, default=3)
    ap.add_argument("--stage-stall-seconds", type=float, default=DEFAULT_STAGE_STALL_SECONDS,
                    help="refuse (by name) when a staged copy makes no byte progress for this long")
    ap.add_argument("--delete-staged-after-consume", action="store_true")
    ap.add_argument("--force-stage", action="store_true",
                    help="stage shards even when the source is a local disk (tests)")
    ap.add_argument("--no-share-read", action="store_true", default=None,
                    help="refuse any read under /mnt/ai_data (the default whenever --src-dir is not there)")
    ap.add_argument("--allow-share-read", action="store_true",
                    help="read the share without the >= 1 MB/s share_rate check, recorded")
    ap.add_argument("--share-rate-receipt", default=DEFAULT_SHARE_RATE_RECEIPT)
    ap.add_argument("--out", default=DEFAULT_OUT, help="big artefacts: dense/, resid/, snapshots/, review/")
    ap.add_argument("--receipt-dir", default=DEFAULT_RECEIPT_DIR)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=12)
    ap.add_argument("--snapshot-keep", type=int, default=2,
                    help="keep the newest N snapshot layers written by this tool; 0 keeps all")
    ap.add_argument("--arm-rung", type=int, default=None)
    ap.add_argument("--arm-layers", default="1-12")
    ap.add_argument("--body-dir", default=None,
                    help="write the non-expert body verbatim here (L{l}/body.safetensors per processed layer "
                         "+ model_level.safetensors when the pass starts at the embedding), byte-checked; "
                         "the OC03 assembly's input")
    ap.add_argument("--row-chunk", type=int, default=1, help="rows per attention call")
    ap.add_argument("--token-chunk", type=int, default=1024, help="tokens per token-wise chunk (apply_attn_res etc.)")
    ap.add_argument("--chunk", type=int, default=1024, help="CoresetFold token chunk")
    ap.add_argument("--holdout-blocks", type=int, default=24, help="ridge check-split blocks over the fit positions")
    ap.add_argument("--max-gram-gb", type=float, default=160.0)
    ap.add_argument("--max-device-gb", type=float, default=-1.0)
    ap.add_argument("--held-rows", dest="held_rows", action="store_true", default=True)
    ap.add_argument("--no-held-rows", dest="held_rows", action="store_false")
    ap.add_argument("--divergence-rows", type=int, default=0, help="cap on rows scored per divergence mask; 0 = all")
    ap.add_argument("--review-topk", type=int, default=16)
    ap.add_argument("--layer0-import-receipt", default=DEFAULT_LAYER0_IMPORT_RECEIPT,
                    help="k3_layer0_import_receipt.json; the verbatim layer-0 tensors must sha-match it "
                         "(its absence is a refusal unless --no-layer0-receipt is given)")
    ap.add_argument("--no-layer0-receipt", action="store_true",
                    help="run without the layer-0 import receipt (synthetic fixtures); recorded in the receipt")
    ap.add_argument("--digest-receipt", default=DEFAULT_DIGEST_RECEIPT)
    ap.add_argument("--chimera-src", default=DEFAULT_CHIMERA_SRC,
                    help="chimera-omni src tree for k3_layer_tensor_names / the dequant cross-check")
    ap.add_argument("--modeling-src", default=DEFAULT_MODELING_SRC, help="K3 modeling files for --mixer-diagnostic")
    ap.add_argument("--mixer-diagnostic", action="store_true")
    ap.add_argument("--acceptance", choices=("gate", "record"), default="gate",
                    help="the measured acceptance rule (G1-G4): 'gate' installs a layer only if it passes "
                         "(default; the real pass); 'record' computes and records the verdict without "
                         "enforcing it (synthetic mechanics tests) -- the receipt says which")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model_dir is None:
        args.model_dir = args.src_dir
    torch.manual_seed(0)
    policy = resolve_share_policy(args.src_dir, no_share_read=args.no_share_read,
                                  allow_share_read=args.allow_share_read,
                                  share_rate_receipt=args.share_rate_receipt)
    args.no_share_read = policy["no_share_read"]
    for path in [args.input_ids] + list(args.review_input_ids):
        if under_share(path) and policy["no_share_read"]:
            raise SystemExit(f"{path} is under {SHARE_ROOT} and share reads are off")
    if not os.path.isdir(args.chimera_src or ""):
        args.chimera_src = None
    index = K3Index(args.src_dir)
    corpus = Corpus(args.input_ids, list(args.review_input_ids))
    stage_enabled = bool(args.stage_dir) and (policy["source_on_share"] or args.force_stage)
    if stage_enabled and not args.delete_staged_after_consume:
        log("WARNING: --stage-dir without --delete-staged-after-consume keeps every consumed shard on disk "
            "(they leave the window but not the directory); the stage dir grows by the whole model")
    stager = ShardStager(args.src_dir, args.stage_dir, enabled=stage_enabled, window=args.stage_window,
                         prefetch=args.prefetch, delete_after_consume=args.delete_staged_after_consume,
                         run_id=uuid.uuid4().hex[:12], stall_seconds=args.stage_stall_seconds)
    run = DualStreamRun(args, index, corpus, stager, policy)
    try:
        run.run()
    finally:
        stager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
