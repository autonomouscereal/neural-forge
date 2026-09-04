#!/usr/bin/env python3
"""Diagnose the K2KV layer-27 gated-writer Gram failure.  Read-only probe.

Three stages, cheapest first, written incrementally to
``logs/probe-K2KV-L27.json`` so a partial run is still evidence:

A  L027 ALONE, kv-only     -- is the pathology local to layer 27's own merge?
B  L027 ALONE, full-group  -- does the other arm fail the same way?
C  L003..L027 sequential, kv-only -- the faithful reproduction of the build.

Nothing is installed, exported or deleted.  The lane's own Harness, capture and
kv_head_merge do the work; this file only wraps them with instruments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

W = "/home/cereal/omni-campaign/qwen-fold"
sys.path.insert(0, W)

import torch  # noqa: E402

import qf_arch  # noqa: E402
import qf_attn  # noqa: E402
import qf_solve  # noqa: E402
from qf_arch import FULL, LINEAR  # noqa: E402
from qf_corpus import fit_check_rows, read_manifest  # noqa: E402
from qf_harness import Harness  # noqa: E402

PARENT = "/home/cereal/models/Qwen3.5-4B"
EVIDENCE = W + "/evidence/omni-evidence-v1.json"
OUT = W + "/logs/probe-K2KV-L27.json"
SUFFIX = ""
TARGET = 27
FULL_POSITIONS = (3, 7, 11, 15, 19, 23, 27)

RESULT: dict = {
    "probe": "K2KV layer-27 gated-writer Gram",
    "parent": PARENT,
    "evidence": EVIDENCE,
    "target_layer": TARGET,
    "device": "cuda:0 (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0)",
    "dtype": "bfloat16",
    "torch": torch.__version__,
    "stages": {},
}


def _jsonable(value):
    """NaN/Inf are not JSON.  Record them as their own names, never as null."""

    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def flush() -> None:
    with open(OUT, "w", encoding="ascii") as handle:
        json.dump(_jsonable(RESULT), handle, indent=2, sort_keys=False,
                  default=str, allow_nan=False)


def say(*parts) -> None:
    stamp = time.strftime("%H:%M:%S")
    print("[probe " + stamp + "]", *parts, flush=True)


# ------------------------------------------------------------------ instruments
def finite_report(t: torch.Tensor) -> dict:
    """Non-finite census of a row block, plus the scale of what IS finite."""

    bad = ~torch.isfinite(t)
    n_bad = int(bad.sum())
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "nonfinite": n_bad,
        "nan": int(torch.isnan(t).sum()),
        "posinf": int(torch.isposinf(t).sum()),
        "neginf": int(torch.isneginf(t).sum()),
    }
    if n_bad and t.dim() == 2:
        per_col = bad.sum(dim=0)
        cols = torch.nonzero(per_col).flatten()
        out["nonfinite_columns"] = int(cols.numel())
        out["first_nonfinite_columns"] = cols[:24].tolist()
        out["nonfinite_rows"] = int(bad.any(dim=1).sum())
        out["first_nonfinite_rows"] = torch.nonzero(bad.any(dim=1)).flatten()[:8].tolist()
    clean = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    out["absmax_finite"] = float(clean.abs().max()) if clean.numel() else 0.0
    return out


def gram_report(A: torch.Tensor, label: str) -> dict:
    """Everything ``prior_anchored_gram`` would have seen, without the crash."""

    report = {"label": label, "rows": int(A.shape[0]), "columns": int(A.shape[1])}
    report["stream"] = finite_report(A)

    clean = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    var = clean.var(dim=0, unbiased=False)
    report["per_channel_variance_min"] = float(var.min())
    report["per_channel_variance_max"] = float(var.max())
    report["per_channel_variance_argmin"] = int(var.argmin())
    report["channels_with_zero_variance"] = int((var == 0).sum())
    report["channels_with_variance_below_1e-30"] = int((var < 1e-30).sum())

    say("  " + label + ": forming Gram from " + str(A.shape[0]) + " rows")
    G = A.T @ A
    diag = torch.diagonal(G)
    report["G_diag_min"] = float(diag.min())
    report["G_diag_max"] = float(diag.max())
    report["G_diag_nonfinite"] = int((~torch.isfinite(diag)).sum())
    report["G_diag_nonpositive"] = int((diag <= 0).sum())
    report["G_diag_argmin"] = int(torch.nan_to_num(diag, nan=float("inf")).argmin())
    report["G_nonfinite_entries"] = int((~torch.isfinite(G)).sum())
    report["G_diag_mean"] = float(diag.mean())
    report["G_diag_mean_is_finite"] = bool(torch.isfinite(diag.mean()))
    report["G_00"] = float(G[0, 0])

    rel = qf_solve.RIDGE_REL
    lam = rel * float(torch.diagonal(G).mean())
    report["ridge_rel"] = rel
    report["lambda_absolute"] = lam
    report["lambda_is_finite"] = bool(lam == lam and abs(lam) != float("inf"))
    Gl = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
    report["Gl_00"] = float(Gl[0, 0])
    try:
        torch.linalg.cholesky(Gl)
        report["cholesky"] = "ok"
    except Exception as error:  # noqa: BLE001
        report["cholesky"] = "FAILED"
        report["cholesky_error"] = type(error).__name__ + ": " + str(error)

    if report["G_nonfinite_entries"] == 0:
        try:
            values = torch.linalg.eigvalsh(0.5 * (G + G.T))
            report["G_eigenvalue_min"] = float(values.min())
            report["G_eigenvalue_max"] = float(values.max())
            report["G_condition"] = float(values.max() / values.min().clamp_min(1e-300))
            report["G_negative_eigenvalues"] = int((values < 0).sum())
        except Exception as error:  # noqa: BLE001
            report["G_eigenvalues"] = "unavailable: " + str(error)
    else:
        report["G_eigenvalues"] = "skipped: Gram carries non-finite entries"
    return report


def weight_report(attn) -> dict:
    out = {}
    for key in ("q_proj", "k_proj", "v_proj", "o_proj"):
        t = getattr(attn, key).weight.detach().to("cpu", torch.float64)
        out[key] = {
            "shape": list(t.shape),
            "absmax": float(t.abs().max()),
            "nonfinite": int((~torch.isfinite(t)).sum()),
        }
    for key in ("q_norm", "k_norm"):
        gain = qf_arch.effective_gain(getattr(attn, key)).detach().to("cpu", torch.float64)
        out[key + "_gain"] = {
            "absmax": float(gain.abs().max()),
            "absmin": float(gain.abs().min()),
            "nonfinite": int((~torch.isfinite(gain)).sum()),
        }
    return out


def submodule_nonfinite(harness, layer_index: int) -> dict:
    """One corpus batch through the child; non-finite census inside layer N."""

    layer = harness.layers()[layer_index]
    attn = layer.self_attn
    seen: dict = {}

    def watch(tag, module):
        def hook(module, args, kwargs, output):
            t = output[0] if isinstance(output, tuple) else output
            t = t.detach()
            seen[tag] = {
                "nonfinite": int((~torch.isfinite(t)).sum()),
                "absmax": float(
                    torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max()
                ),
                "shape": list(t.shape),
            }
            return None

        return module.register_forward_hook(hook, with_kwargs=True)

    def watch_input(tag, module):
        def hook(module, args, kwargs):
            t = (args[0] if args else kwargs["hidden_states"]).detach()
            seen[tag] = {
                "nonfinite": int((~torch.isfinite(t)).sum()),
                "absmax": float(
                    torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max()
                ),
                "shape": list(t.shape),
            }
            return None

        return module.register_forward_pre_hook(hook, with_kwargs=True)

    handles = [
        watch_input("layer_input_hidden", layer),
        watch_input("attn_input_hidden", attn),
        watch("q_proj_out", attn.q_proj),
        watch("k_proj_out", attn.k_proj),
        watch("v_proj_out", attn.v_proj),
        watch_input("o_proj_input_gated", attn.o_proj),
        watch("o_proj_out", attn.o_proj),
    ]
    try:
        with torch.no_grad():
            harness.model(input_ids=harness.batches[0][1].to(harness.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return seen


# ---------------------------------------------------------------- patch plumbing
def restrict_positions(harness, positions) -> None:
    """Make :func:`qf_attn.kv_head_merge` see only ``positions`` as full-attention.

    Matched by module IDENTITY, not by ``layer_idx``: the merge enumerates the
    live layer list, and an attribute that a future transformers release renames
    would silently widen the probe to every full-attention site.
    """

    wanted = {id(layer) for i, layer in enumerate(harness.layers()) if i in set(positions)}
    original = qf_arch.layer_kind

    def patched(layer):
        kind = original(layer)
        if kind != FULL:
            return kind
        return FULL if id(layer) in wanted else LINEAR

    qf_arch.layer_kind = patched
    qf_attn.qf_arch.layer_kind = patched


class _StopProbe(Exception):
    """Raised after the target layer is diagnosed; ends the merge early."""


def instrument_resolver(bucket: dict, target: int) -> None:
    """Wrap ``_resolve_writer``: diagnose the target layer, pass the rest through."""

    original = qf_attn._resolve_writer
    name_wanted = "L%03d" % target

    # **kwargs, not a fixed arity.  The first after-fix run of this probe burned
    # 40 minutes of GPU raising TypeError because qf_attn._resolve_writer had
    # gained a `log=` keyword and this monkeypatch had not: a probe whose
    # signature is pinned to the function it wraps breaks every time that
    # function is the thing being fixed.
    def patched(harness, attn, name, Y, Y_hld, lambda_split, dtype, **extra):
        if name != name_wanted:
            return original(harness, attn, name, Y, Y_hld, lambda_split, dtype, **extra)

        entry: dict = {"layer": name, "installed_child_weights": weight_report(attn)}
        entry["target_local_fit"] = finite_report(Y)
        say("  " + name + ": capturing the child's own post-gate writer stream")
        started = time.time()
        stream = harness.capture({name: attn.o_proj})[name]
        entry["capture_seconds"] = round(time.time() - started, 1)
        A = stream["fit"]

        check = fit_check_rows(harness.manifest) if lambda_split else None
        entry["lambda_split"] = bool(lambda_split)
        if check is not None:
            inner = ~check
            entry["gram_inner_split"] = gram_report(A[inner], name + " G_in (inner split)")
        entry["gram_all_rows"] = gram_report(A, name + " G_all (all fit rows)")

        say("  " + name + ": one-batch non-finite census inside the child layer")
        try:
            entry["submodule_census_one_batch"] = submodule_nonfinite(harness, target)
        except Exception as error:  # noqa: BLE001
            entry["submodule_census_one_batch"] = "unavailable: " + str(error)

        # qf_attn.attention_gate_probe is deliberately NOT called here: at
        # sequence_length 4096 its fp64 [1, 16, 4096, 4096] logits are 8 GB of
        # device memory on top of a 9.3 GB model, and an OOM inside the probe
        # would destroy the state we came to measure.  The submodule census
        # above localises a non-finite to a projection without it.

        prior = attn.o_proj.weight.detach().to("cpu", torch.float64)
        try:
            writer, guard = qf_solve.gated_out_proj_solve(A, Y, prior, check)
            entry["solve"] = {"outcome": "ok", "guard": guard}
        except Exception as error:  # noqa: BLE001
            entry["solve"] = {
                "outcome": "FAILED",
                "error": type(error).__name__ + ": " + str(error),
            }
        bucket["target"] = entry
        raise _StopProbe(name)

    qf_attn._resolve_writer = patched


def fresh_harness(manifest) -> Harness:
    return Harness(PARENT, "cuda:0", torch.bfloat16, manifest)


def run_stage(key: str, label: str, positions, arm: str, manifest) -> None:
    say("STAGE " + key + ": " + label)
    started = time.time()
    key = key + SUFFIX
    stage: dict = {"label": label, "arm": arm, "positions": list(positions)}
    RESULT["stages"][key] = stage
    flush()

    saved_kind = qf_arch.layer_kind
    saved_resolver = qf_attn._resolve_writer
    harness = None
    try:
        harness = fresh_harness(manifest)
        restrict_positions(harness, positions)
        bucket: dict = {}
        instrument_resolver(bucket, TARGET)
        try:
            qf_attn.kv_head_merge(
                harness, kv_heads=2, arm=arm, lambda_split=True,
                dtype=torch.bfloat16, log=lambda line: say("   ", line),
            )
            stage["merge"] = "completed without reaching the probe stop"
        except _StopProbe as stop:
            stage["merge"] = "stopped at " + str(stop) + " after diagnosis (by design)"
        stage.update(bucket.get("target", {"target": "never reached"}))
    except Exception as error:  # noqa: BLE001
        stage["stage_error"] = type(error).__name__ + ": " + str(error)
        stage["traceback"] = traceback.format_exc()
    finally:
        qf_arch.layer_kind = saved_kind
        qf_attn.qf_arch.layer_kind = saved_kind
        qf_attn._resolve_writer = saved_resolver
        if harness is not None:
            del harness
        torch.cuda.empty_cache()
    stage["seconds"] = round(time.time() - started, 1)
    flush()
    say("STAGE " + key + " done in " + str(stage["seconds"]) + "s")


def main() -> int:
    global OUT, SUFFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="A,B,C")
    parser.add_argument("--out", default=OUT)
    parser.add_argument(
        "--suffix", default="",
        help="appended to each stage key, so a before/after pair can be merged",
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    OUT = args.out
    SUFFIX = args.suffix
    if args.label:
        RESULT["run_label"] = args.label
    wanted = {s.strip().upper() for s in args.stages.split(",") if s.strip()}

    manifest = read_manifest(EVIDENCE)
    RESULT["evidence_manifest_sha256"] = manifest["manifest_sha256"]
    RESULT["fit_rows"] = sum(
        len(r["ids"]) for r in manifest["rows"] if r["split"] == "fit"
    )
    RESULT["internal_check_rows"] = int(fit_check_rows(manifest).sum())
    flush()

    if "A" in wanted:
        run_stage("A", "layer 27 alone, kv-only", (TARGET,), "kv-only", manifest)
    if "B" in wanted:
        run_stage("B", "layer 27 alone, full-group", (TARGET,), "full-group", manifest)
    if "C" in wanted:
        run_stage(
            "C", "layers 3..27 sequential, kv-only (faithful reproduction)",
            FULL_POSITIONS, "kv-only", manifest,
        )
    flush()
    say("probe complete -> " + OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
