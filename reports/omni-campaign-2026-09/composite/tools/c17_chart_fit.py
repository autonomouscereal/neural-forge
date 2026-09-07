#!/usr/bin/env python
"""c17_chart_fit.py -- S4 cross-width charts + 92 -> 64 depth alignment (run-book row C17).

For every Chimera trunk layer ``k`` this tool fits closed-form charts between the K3
folded-stream FFN-input rows (K3 hidden, 7,168 wide) and the trunk's FFN-input rows
(5,120 wide) on the SAME held text, paired token-by-token through the S1
``boundary_map.json`` (Qwen <-> K3 position pairs at common byte boundaries):

    C_in  [d_k3   x d_trunk]   y_hat = x_k3    @ C_in  + C_in_bias     (K3 -> trunk)
    C_out [d_trunk x d_k3  ]   x_hat = y_trunk @ C_out + C_out_bias    (trunk -> K3)

Three arms per trunk layer, all float64 on the CPU (``mediafold.linalg``):

* ``ridge``       -- ridge over the lambda ladder (the chart of record; C_in / C_out are
                     fitted independently, fit rows only, lambda chosen on an inner split of
                     the fit rows, then refitted on every fit row);
* ``svd``         -- orthonormal SVD chart of the CENTRED K3 fit rows at rank d_trunk
                     (control: C_out = C_in^T, no knowledge of the trunk).  The row mean is
                     folded into the biases (C_in_bias = -mean_x C_in, C_out_bias = mean_x) so
                     the round-trip reconstructs the mean component instead of dropping it;
* ``procrustes``  -- orthogonal Procrustes between the CENTRED K3 rows expressed in that SVD
                     chart and the CENTRED trunk rows (plain Procrustes when the widths are
                     equal); C_in has orthonormal columns, C_out = C_in^T, and both means are
                     folded into the biases (C_in_bias = mean_y - mean_x C_in, C_out_bias =
                     mean_x - mean_y C_out).  Residual streams at the FFN input carry large
                     fixed offsets: an uncentred control is dominated by the offset.

Depth alignment (``--alignment 92:64 --dense-layer0 trunk0``, the floor-interval map):
trunk layer ``k`` receives K3 MoE layers ``[1 + floor(92 k / 64), 1 + floor(92 (k+1) / 64))``
-- 28 trunk layers receive two MoE layers, 36 receive one -- and the dense K3 layer 0 is
added to trunk layer 0's pool.  Every K3 layer 0..92 lands in exactly one trunk pool.
Where a pool holds several K3 layers their paired rows are concatenated (the trunk row
repeats once per K3 layer) and the held-out split is drawn on TRUNK TOKENS, so no token
sits on both sides of the split in any arm.

Inputs
------
``--k3-resid DIR``     ``DIR/L{l:02d}/held_rows.safetensors`` written by ``k3_dense_pass.py``
                       (``held_positions`` int64 [rows] = flat token index ``row * T + pos``
                       into the K3 evidence ids file; ``folded_moe_x`` / ``teacher_moe_x``
                       [rows, 7168]; metadata ``input_ids_sha256`` = the pass's corpus sha).
                       ``--k3-stream folded`` (default) charts the FOLDED stream -- the rows
                       the union install will actually see.
``--k3-pass-receipt``  the ``k3_dense_pass_receipt.json`` of that pass (default: searched
                       beside ``--k3-resid`` and under the pass's default receipt dir).  Its
                       ``corpus.corpus_sha256`` must equal every held_rows file's
                       ``input_ids_sha256`` metadata, and its evidence segment's
                       ``input_ids_sha256`` (sha256 over the int64 ids bytes) must equal the
                       boundary map's ``ids_content_sha256[--boundary-file].int64`` -- this
                       is what ties the K3 rows to the boundary map's ids.  Missing receipt =
                       refusal unless ``--allow-unverified-k3-corpus`` (recorded as such).
``--trunk-taps DIR``   the trunk tap run's output dir (``run_omni_frozen_review.py --stream
                       --teacher-forced --tap-residuals --tap-out DIR --input <evidence
                       manifest> --held-only``).  Two layouts are accepted per trunk layer:

                       * ``DIR/L{k:02d}/ffn_input_rows.safetensors`` -- what the runner
                         WRITES (schema ``chimera-omni-ffn-input-taps-v1``): ``prenorm``
                         [rows, 5120] = the input of ``blocks.k.norm2`` (the FFN input,
                         post-attention residual; ``normed`` on request) and NO positions.
                         Their token positions are reconstructed from ``DIR/taps_receipt.json``
                         (``stride``, ``evidence.rows`` = tapped ``batch_index`` list) plus the
                         evidence manifest (``--evidence-manifest``, default the path the taps
                         receipt names, sha256-checked) by replaying the runner's selection rule
                         (sequences sorted by descending length into micro-batches of
                         ``--tap-micro-batch`` = 8, every ``stride``-th valid token in row-major
                         order under one global counter); the row count must match exactly and
                         the file's sha256 must match the receipt's ``layers[].sha256``.
                       * ``DIR/L{k:02d}/held_rows.safetensors`` -- ``held_positions`` int64
                         [rows] = flat ``row * T + pos`` into the Qwen evidence ids, or
                         ``positions`` int64 [rows, 2] = (row_index, position), plus rows under
                         one of ``--trunk-rows-key`` (auto-detected among ``prenorm``, ``ffn_in``,
                         ``mlp_in``, ``moe_x``, ``teacher_moe_x``, ``folded_moe_x``).

                       Row order everywhere: ``omni-evidence-27b-v1.json`` rows sorted by
                       ``batch_index`` == ``qwen_input_ids.safetensors`` == ``boundary_map.json``
                       ``row_index`` (k3_corpus_build iterates the manifest that way).
``--boundary-map``     S1 ``boundary_map.json`` (schema ``neural-forge-k3-boundary-map-v2``).
                       Its ``boundary_map_sha256`` seal is RECOMPUTED (estate canonical JSON
                       over the body minus the seal) and must match; with a taps receipt its
                       ``evidence_manifest_file_sha256`` must equal the taps receipt's
                       ``evidence.manifest_sha256`` (manifest <-> boundary map <-> taps closed).

Outputs (``--out``)
-------------------
``L{k:02d}/ridge.safetensors``, ``L{k:02d}/svd_chart.safetensors``,
``L{k:02d}/procrustes.safetensors`` (each ``C_in``, ``C_in_bias``, ``C_out``, ``C_out_bias``
in float32 unless ``--chart-dtype float64``; the two control charts are skipped on
``--skip-control-charts``, their numbers stay in the receipt), ``L{k:02d}/layer.json``,
``alignment.json``, ``receipt.json`` (schema ``RECEIPT_SCHEMA``; per layer: K3 layers,
paired rows per K3 layer, unique trunk tokens, the gate quantity, fit / held counts, Gram
diagnostics, ridge design regime per direction, held-out cos / relL2 of every arm and
direction, output sha256s, status fitted / reused; the receipt's own sha256 is printed last).

Footprint: three fp32 arms at 7,168 x 5,120 are 3 x 293.65 MB per layer = 56.4 GB over 64
layers.  The projected bytes are printed before the first write and ``--out`` under ``/home``
(the root disk) is refused unless ``--allow-root-disk``: large artefacts live under
``/mnt/lexar/omni-campaign/<lane>/``.

Resume / merge: a requested layer whose ``L{k}/layer.json`` + arm files verify by sha256
against unchanged inputs and the same fit signature is REUSED (``--refit`` forces a refit);
``receipt.json`` is rebuilt from EVERY ``L*/layer.json`` present, each marked ``fitted``
(this invocation), ``reused`` or ``invalid``, so a chunked or restarted 64-layer run never
leaves a partial receipt beside charts it does not describe.

Refusals
--------
* a trunk layer whose gate quantity -- ``min(unique trunk tokens, min paired rows over its
  K3 layers)``, NOT the pooled row count -- is below ``--min-paired-rows`` (default 4,096) is
  refused unless ``--allow-underdetermined`` (then fitted and flagged ``underdetermined``);
* a non-finite Gram (NaN / Inf rows) is a named refusal, never a silent fallback;
* an unknown ``--dense-layer0`` policy or a pair outside the sequence length is refused;
* a positions-free tap dump whose row count differs from the replayed selection, whose sha256
  differs from the taps receipt's, or whose taps receipt names a manifest with another
  sha256, is refused (never guessed);
* any K3 / boundary-map / taps identity mismatch (above) is refused.

``mediafold`` is bound from ``../../media`` beside this tool (both trees keep that layout)
or from ``OMNI_MEDIA_ROOT``.

Section 3b: nothing here divides by a norm gain; structurally dead input channels (the
27B's exact-zero ``post_attention_layernorm`` gain at layer 7 feeds the MLP input) are
COUNTED per side as ``dead_columns`` and handled by the ridge, never treated as failures.
The run book's ">= 4,096 rows is over-determined" line is false at 7,168 wide: the receipt
records per direction whether the ladder's inner fit and the final refit ran with fewer rows
than design columns (``ladder_regime`` / ``refit_regime``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

TOOL_NAME = "c17_chart_fit"
TOOL_VERSION = "c17-chart-fit-v1.3"
RECEIPT_SCHEMA = "chimera_omni.composite.c17_chart_fit.receipt.v1"
DEFAULT_MIN_PAIRED_ROWS = 4096
DEFAULT_ALIGNMENT = "92:64"
DENSE_LAYER0_POLICIES = ("trunk0", "skip")
ARMS = ("ridge", "svd", "procrustes")
CONTROL_ARMS = ("svd", "procrustes")
ARM_FILES = {"ridge": "ridge.safetensors", "svd": "svd_chart.safetensors", "procrustes": "procrustes.safetensors"}
TRUNK_ROW_KEYS = ("prenorm", "ffn_in", "mlp_in", "moe_x", "teacher_moe_x", "folded_moe_x")
TRUNK_FILE_NAMES = ("ffn_input_rows.safetensors", "held_rows.safetensors")
TAP_SCHEMA = "chimera-omni-ffn-input-taps-v1"  # run_omni_frozen_review.TAP_SCHEMA
TAP_RECEIPT_NAME = "taps_receipt.json"
RUNNER_MICRO_BATCH = 8  # run_omni_frozen_review.StreamedChimera.run(micro_batch=8)
K3_STREAM_KEYS = {"folded": "folded_moe_x", "teacher": "teacher_moe_x"}
BOUNDARY_SCHEMA = "neural-forge-k3-boundary-map-v2"
BOUNDARY_SEAL_FIELD = "boundary_map_sha256"
K3_PASS_RECEIPT_SCHEMA = "omni-campaign.k3_dense_pass.v1"  # k3_dense_pass.RECEIPT_SCHEMA
K3_PASS_RECEIPT_NAMES = ("k3_dense_pass_receipt.json", "k3_dense_pass_receipt.partial.json")
DEFAULT_K3_PASS_RECEIPT_DIR = "/home/cereal/omni-campaign/composite/receipts/k3_dense_pass"
ROOT_DISK_PREFIX = "/home"
SAFETENSORS_HEADER_ALLOWANCE = 512  # bytes; the 4-tensor header is ~300 bytes
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# lane binding: mediafold (media/mediafold beside composite/tools on both trees)
# --------------------------------------------------------------------------------------


def resolve_media_root(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("OMNI_MEDIA_ROOT"):
        candidates.append(Path(os.environ["OMNI_MEDIA_ROOT"]))
    candidates.append(HERE.parent.parent / "media")
    candidates.append(Path("/home/cereal/omni-campaign/media"))
    for root in candidates:
        if (root / "mediafold" / "linalg.py").is_file():
            return root
    raise SystemExit(f"mediafold not found; tried {[str(c) for c in candidates]} (set OMNI_MEDIA_ROOT)")


def bind_mediafold(explicit: str | None = None):
    root = resolve_media_root(explicit)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mediafold import linalg, receipts  # noqa: E402

    return root, linalg, receipts


MEDIA_ROOT, linalg, receipts = bind_mediafold(os.environ.get("OMNI_MEDIA_ROOT"))


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str]) -> str:
    return receipts.hash_file(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def estate_canonical_sha256(value: Any) -> str:
    """The estate's canonical serialisation (``qf_arch.canonical_json_bytes``: ascii, sorted
    keys, no spaces, no NaN) -- the exact bytes ``k3_corpus_build.write_json`` seals
    ``boundary_map_sha256`` over.  Do not "improve" this."""

    return hashlib.sha256(json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"),
                                     sort_keys=True).encode("ascii")).hexdigest()


def read_metadata(path: str) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def under_root_disk(path: str | os.PathLike[str]) -> bool:
    """True when ``path`` resolves under ``/home`` (the server's root disk)."""

    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(path).absolute()
    text = resolved.as_posix()
    return text == ROOT_DISK_PREFIX or text.startswith(ROOT_DISK_PREFIX + "/")


def projected_chart_bytes(d_x: int, d_y: int, chart_dtype: torch.dtype, arms_written: Sequence[str],
                          layers: int) -> dict[str, Any]:
    """Bytes one arm file takes at these widths, per layer over the written arms, and in total."""

    itemsize = torch.tensor([], dtype=chart_dtype).element_size()
    per_arm = (2 * d_x * d_y + d_x + d_y) * itemsize + 8 + SAFETENSORS_HEADER_ALLOWANCE
    per_layer = per_arm * len(arms_written)
    return {
        "d_k3": int(d_x), "d_trunk": int(d_y), "chart_dtype": str(chart_dtype), "arms_written": list(arms_written),
        "bytes_per_arm": int(per_arm), "bytes_per_layer": int(per_layer), "layers": int(layers),
        "bytes_total": int(per_layer * layers),
    }


# --------------------------------------------------------------------------------------
# depth alignment
# --------------------------------------------------------------------------------------


def parse_alignment(text: str) -> tuple[int, int]:
    try:
        left, right = text.split(":")
        moe_layers, trunk_layers = int(left), int(right)
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"--alignment must be <k3_moe_layers>:<trunk_layers>, got {text!r}") from error
    if moe_layers < 1 or trunk_layers < 1:
        raise SystemExit(f"--alignment needs positive counts, got {text!r}")
    if moe_layers < trunk_layers:
        raise SystemExit(f"--alignment {text!r}: fewer K3 MoE layers than trunk layers is not a depth merge")
    return moe_layers, trunk_layers


def alignment_table(moe_layers: int, trunk_layers: int, dense_layer0: str = "trunk0") -> list[dict[str, Any]]:
    """Floor-interval map of K3 MoE layers 1..moe_layers onto trunk layers 0..trunk_layers-1.

    Trunk ``k`` receives MoE layers ``[1 + floor(M k / N), 1 + floor(M (k+1) / N))``.  The
    dense K3 layer 0 goes to trunk 0 (``trunk0``) or nowhere (``skip``).  Returned rows carry
    ``k3_layers`` (what is pooled, including the dense layer) and ``moe_layers`` (the
    routed layers only, the count the run book quotes: 28 twos and 36 ones at 92:64).
    """

    if dense_layer0 not in DENSE_LAYER0_POLICIES:
        raise SystemExit(f"--dense-layer0 must be one of {DENSE_LAYER0_POLICIES}, got {dense_layer0!r}")
    table: list[dict[str, Any]] = []
    for k in range(trunk_layers):
        start = 1 + (moe_layers * k) // trunk_layers
        stop = 1 + (moe_layers * (k + 1)) // trunk_layers
        moe = list(range(start, stop))
        k3 = list(moe)
        if k == 0 and dense_layer0 == "trunk0":
            k3 = [0] + k3
        table.append({"trunk": k, "k3_layers": k3, "moe_layers": moe, "moe_start": start, "moe_stop": stop})
    return table


def alignment_summary(table: list[dict[str, Any]], moe_layers: int, dense_layer0: str) -> dict[str, Any]:
    covered = [layer for row in table for layer in row["k3_layers"]]
    expected = list(range(0 if dense_layer0 == "trunk0" else 1, moe_layers + 1))
    counts = [len(row["moe_layers"]) for row in table]
    return {
        "k3_moe_layers": moe_layers,
        "trunk_layers": len(table),
        "dense_layer0": dense_layer0,
        "k3_layers_covered": len(covered),
        "covered_exactly_once": sorted(covered) == expected and len(covered) == len(set(covered)),
        "trunk_layers_with_two_moe_layers": sum(1 for c in counts if c == 2),
        "trunk_layers_with_one_moe_layer": sum(1 for c in counts if c == 1),
        "trunk_layers_with_other_counts": sum(1 for c in counts if c not in (1, 2)),
        "rule": "trunk k <- K3 MoE layers [1 + floor(M k / N), 1 + floor(M (k+1) / N)); dense layer 0 -> "
                + ("trunk 0" if dense_layer0 == "trunk0" else "nowhere"),
    }


# --------------------------------------------------------------------------------------
# row loading and pairing
# --------------------------------------------------------------------------------------


def load_rows_file(path: str, row_keys: Sequence[str], sequence_length: int | None,
                   fallback_positions: torch.Tensor | None = None) -> dict[str, Any]:
    """Load one rows file: rows (float64) + flat positions (int64).

    Positions come from the file (``held_positions`` flat, or ``positions`` [rows, 2]); a
    positions-free dump (the runner's ``ffn_input_rows.safetensors``) takes
    ``fallback_positions`` (replayed from the taps receipt) and its row count must match.
    """

    from safetensors.torch import load_file

    if not os.path.isfile(path):
        raise SystemExit(f"missing rows file {path}")
    got = load_file(path)
    key = next((k for k in row_keys if k in got), None)
    if key is None:
        raise SystemExit(f"{path}: none of the row keys {list(row_keys)} present (has {sorted(got)})")
    rows = got[key]
    if rows.dim() != 2:
        raise SystemExit(f"{path}[{key}]: rows must be [rows, width], got {list(rows.shape)}")
    if "held_positions" in got:
        positions = got["held_positions"].to(torch.int64).flatten()
        position_key = "held_positions"
    elif "positions" in got:
        pos = got["positions"].to(torch.int64)
        if pos.dim() != 2 or pos.shape[1] != 2:
            raise SystemExit(f"{path}[positions]: expected [rows, 2] = (row_index, position)")
        if sequence_length is None:
            raise SystemExit(f"{path}: (row, position) pairs need a sequence length to flatten")
        positions = pos[:, 0] * int(sequence_length) + pos[:, 1]
        position_key = "positions"
    elif fallback_positions is not None:
        positions = fallback_positions.to(torch.int64).flatten()
        position_key = "replayed:taps_receipt"
        if positions.numel() != rows.shape[0]:
            raise SystemExit(
                f"{path}: {rows.shape[0]} rows but the replayed tap selection yields {positions.numel()} positions; "
                f"the dump was not produced by the selection the taps receipt describes -- refusing"
            )
    else:
        raise SystemExit(f"{path}: no held_positions / positions tensor (and no taps_receipt.json to replay)")
    if positions.numel() != rows.shape[0]:
        raise SystemExit(f"{path}: {positions.numel()} positions for {rows.shape[0]} rows")
    if positions.numel() != int(torch.unique(positions).numel()):
        raise SystemExit(f"{path}: duplicate positions")
    if bool((positions < 0).any()):
        raise SystemExit(f"{path}: negative positions")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "rows_key": key,
        "position_key": position_key,
        "dtype": str(rows.dtype),
        "rows": rows.to(torch.float64),
        "positions": positions,
        "metadata": read_metadata(path),
    }


def resolve_trunk_file(trunk_dir: str, k: int, explicit: str | None = None) -> str:
    """``L{k:02d}/<name>``: the explicit name, else the first of TRUNK_FILE_NAMES present."""

    layer_dir = os.path.join(trunk_dir, f"L{k:02d}")
    names = (explicit,) if explicit else TRUNK_FILE_NAMES
    for name in names:
        candidate = os.path.join(layer_dir, name)
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit(f"trunk layer {k}: none of {list(names)} under {layer_dir}")


def load_taps_receipt(trunk_dir: str) -> dict[str, Any] | None:
    """The runner's ``taps_receipt.json`` beside the ``L*/`` dirs, or None when absent."""

    path = os.path.join(trunk_dir, TAP_RECEIPT_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        body = json.load(handle)
    if body.get("schema") != TAP_SCHEMA:
        raise SystemExit(f"{path}: schema {body.get('schema')!r} != {TAP_SCHEMA!r}")
    return {"path": path, "sha256": sha256_file(path), "body": body}


def taps_layer_entry(taps: dict[str, Any] | None, k: int) -> dict[str, Any] | None:
    if taps is None:
        return None
    for entry in taps["body"].get("layers", []):
        if int(entry.get("layer", -1)) == int(k):
            return entry
    return None


def check_tap_file_sha(trunk: dict[str, Any], taps: dict[str, Any] | None, k: int) -> bool | None:
    """The runner records ``layers[].sha256`` per tap file; the file actually loaded must match.
    Returns True (matched), None (nothing recorded to compare) or refuses."""

    entry = taps_layer_entry(taps, k)
    recorded = (entry or {}).get("sha256")
    if not recorded:
        return None
    if str(recorded) != trunk["sha256"]:
        raise SystemExit(f"{trunk['path']}: sha256 {trunk['sha256'][:16]} != the taps receipt's layers[{k}].sha256 "
                         f"{str(recorded)[:16]}; the file is not the one the runner wrote -- refusing")
    return True


def replay_tap_selection(
    manifest_rows: Sequence[dict[str, Any]],
    tapped_batch_indices: Sequence[int],
    *,
    stride: int,
    micro_batch: int,
    sequence_length: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Replay ``run_omni_frozen_review``'s evidence tap selection and return the flat
    positions (``row_index * T + pos``) of the dumped rows, in dump order.

    The runner (``evidence_sequences`` + ``StreamedChimera.run`` + the ``select`` closure):
    sequences = manifest rows sorted by ``batch_index`` (filtered to the tapped ones),
    ``lengths[i] = len(ids)``; ``order`` = indices sorted by descending length (stable);
    micro-batches of ``micro_batch`` in that order; per batch the validity mask is
    ``valid[row, :length]``; the selection walks ``nonzero(valid.reshape(-1))`` (row-major)
    keeping every ``stride``-th valid token under ONE counter carried across batches; the
    hook stores ``x[mask]`` (row-major) per batch and the layer file is the concatenation
    over batches.  ``row_index`` = the row's rank among ALL manifest rows sorted by
    ``batch_index`` (k3_corpus_build's row order == boundary_map row_index).
    """

    if int(stride) < 1 or int(micro_batch) < 1:
        raise SystemExit(f"tap replay needs stride >= 1 and micro_batch >= 1, got {stride} / {micro_batch}")
    ordered = sorted(manifest_rows, key=lambda r: int(r["batch_index"]))
    row_index_of = {int(r["batch_index"]): i for i, r in enumerate(ordered)}
    length_of = {int(r["batch_index"]): len(r["ids"]) for r in ordered}
    if len(row_index_of) != len(ordered):
        raise SystemExit("evidence manifest: duplicate batch_index")
    tapped = [int(b) for b in tapped_batch_indices]
    if tapped != sorted(tapped):
        raise SystemExit("taps receipt evidence.rows are not in ascending batch_index order; not the runner's order")
    missing = [b for b in tapped if b not in row_index_of]
    if missing:
        raise SystemExit(f"taps receipt names batch_index {missing} absent from the evidence manifest")
    lengths = [length_of[b] for b in tapped]
    too_long = [b for b, n in zip(tapped, lengths) if n > int(sequence_length)]
    if too_long:
        raise SystemExit(f"manifest rows {too_long} exceed the sequence length {sequence_length}")
    order = sorted(range(len(tapped)), key=lambda i: -lengths[i])
    batches = [order[i: i + int(micro_batch)] for i in range(0, len(order), int(micro_batch))]
    flat: list[int] = []
    counter = 0
    for batch in batches:
        for i in batch:
            base = row_index_of[tapped[i]] * int(sequence_length)
            for p in range(lengths[i]):
                if counter % int(stride) == 0:
                    flat.append(base + p)
                counter += 1
    return torch.tensor(flat, dtype=torch.int64), {
        "stride": int(stride), "micro_batch": int(micro_batch), "sequence_length": int(sequence_length),
        "tapped_batch_indices": tapped, "tapped_rows": len(tapped), "valid_tokens": counter,
        "selected_rows": len(flat), "batches": len(batches),
        "rule": "run_omni_frozen_review: sequences sorted by descending length into micro-batches; every stride-th "
                "valid token in row-major order under one global counter; rows concatenated over batches",
    }


def replayed_trunk_positions(trunk_dir: str, taps: dict[str, Any], manifest_path: str | None,
                             *, micro_batch: int, sequence_length: int) -> dict[str, Any]:
    """Positions for a positions-free tap dump: taps receipt + evidence manifest -> flat positions."""

    body = taps["body"]
    evidence = body.get("evidence") or {}
    if "stride" not in body or "rows" not in evidence:
        raise SystemExit(f"{taps['path']}: not an evidence tap receipt (needs stride + evidence.rows); "
                         f"rows_from = {body.get('rows_from')!r}")
    path = manifest_path or evidence.get("manifest")
    if not path or not os.path.isfile(path):
        raise SystemExit(f"evidence manifest {path!r} not found; pass --evidence-manifest")
    manifest_sha = sha256_file(path)
    recorded = evidence.get("manifest_sha256")
    if recorded and recorded != manifest_sha:
        raise SystemExit(f"evidence manifest {path} sha256 {manifest_sha[:16]} != the taps receipt's "
                         f"{str(recorded)[:16]}; the taps were not drawn from this manifest -- refusing")
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    positions, notes = replay_tap_selection(manifest["rows"], evidence["rows"], stride=int(body["stride"]),
                                            micro_batch=micro_batch, sequence_length=sequence_length)
    per_layer = {int(entry["layer"]): int(entry["rows"]) for entry in body.get("layers", [])}
    off = {layer: n for layer, n in per_layer.items() if n != positions.numel()}
    if off:
        raise SystemExit(f"{taps['path']}: layers {sorted(off)} record {sorted(set(off.values()))} rows but the "
                         f"replayed selection yields {positions.numel()}; refusing")
    return {
        "positions": positions,
        "source": "replayed from taps_receipt.json + evidence manifest",
        "taps_receipt": {"path": taps["path"], "sha256": taps["sha256"], "rows_from": body.get("rows_from"),
                         "stride": int(body["stride"]), "tap_max_rows": body.get("tap_max_rows"),
                         "held_only": evidence.get("held_only"), "manifest_sha256": recorded},
        "evidence_manifest": {"path": path, "sha256": manifest_sha,
                              "sha256_verified_against_taps_receipt": bool(recorded)},
        "replay": notes,
    }


def load_boundary_map(path: str, boundary_file: str) -> dict[str, Any]:
    """Load + verify the S1 boundary map: schema, the recomputed ``boundary_map_sha256`` seal
    (estate canonical JSON over the body minus the seal, ``k3_corpus_build.write_json``), and
    the rows of ``boundary_file``."""

    with open(path, "r", encoding="utf-8") as handle:
        body = json.load(handle)
    schema = body.get("schema")
    if schema != BOUNDARY_SCHEMA:
        raise SystemExit(f"{path}: schema {schema!r} != {BOUNDARY_SCHEMA!r}")
    recorded_seal = body.get(BOUNDARY_SEAL_FIELD)
    if not recorded_seal:
        raise SystemExit(f"{path}: no {BOUNDARY_SEAL_FIELD} seal; an unsealed boundary map cannot be tied to its ids")
    recomputed_seal = estate_canonical_sha256({k: v for k, v in body.items() if k != BOUNDARY_SEAL_FIELD})
    if str(recorded_seal) != recomputed_seal:
        raise SystemExit(f"{path}: {BOUNDARY_SEAL_FIELD} {str(recorded_seal)[:16]} != recomputed {recomputed_seal[:16]}; "
                         "the boundary map was edited after k3_corpus_build sealed it -- refusing")
    rows = [r for r in body.get("rows", []) if r.get("file") == boundary_file]
    if not rows:
        raise SystemExit(f"{path}: no rows for file {boundary_file!r}")
    seen = set()
    for row in rows:
        if row["row_index"] in seen:
            raise SystemExit(f"{path}: duplicate row_index {row['row_index']} for {boundary_file}")
        seen.add(row["row_index"])
    return {
        "path": path,
        "sha256": sha256_file(path),
        "boundary_map_sha256": str(recorded_seal),
        "boundary_map_sha256_recomputed": recomputed_seal,
        "boundary_map_seal_verified": True,
        "schema": schema,
        "sequence_length": int(body["sequence_length"]),
        "file": boundary_file,
        "rows": rows,
        "aggregate": body.get("aggregate"),
        "ids_content_sha256": body.get("ids_content_sha256"),
        "evidence_manifest_sha256": body.get("evidence_manifest_sha256"),
        "evidence_manifest_file_sha256": body.get("evidence_manifest_file_sha256"),
        "evidence_manifest_path": body.get("evidence_manifest_path"),
    }


def boundary_ids_int64_sha256(boundary: dict[str, Any]) -> str | None:
    """``ids_content_sha256[file]`` as k3_corpus_build writes it (``{"int64": ..., "fp32": ...}``,
    the sha256 over the raw int64 ids bytes) or a bare string from an older writer."""

    table = boundary.get("ids_content_sha256") or {}
    value = table.get(boundary["file"])
    if isinstance(value, dict):
        value = value.get("int64")
    return str(value) if value else None


def load_k3_pass_receipt(explicit: str | None, k3_resid: str) -> dict[str, Any] | None:
    """The ``k3_dense_pass`` receipt the resid/ rows came from: explicit path, else searched
    beside ``--k3-resid`` (``<resid>/..``) and under the pass's default receipt dir; the final
    receipt before the ``.partial.json`` one.  None when nothing is found."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        for base in (Path(k3_resid).resolve().parent, Path(k3_resid).resolve().parent / "receipts",
                     Path(DEFAULT_K3_PASS_RECEIPT_DIR)):
            for name in K3_PASS_RECEIPT_NAMES:
                candidates.append(base / name)
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as handle:
                body = json.load(handle)
            if body.get("schema") != K3_PASS_RECEIPT_SCHEMA:
                raise SystemExit(f"{candidate}: schema {body.get('schema')!r} != {K3_PASS_RECEIPT_SCHEMA!r}")
            corpus = body.get("corpus") or {}
            segments = corpus.get("segments") or []
            if not corpus.get("corpus_sha256") or not segments:
                raise SystemExit(f"{candidate}: no corpus.corpus_sha256 / corpus.segments; not a k3_dense_pass receipt")
            evidence = segments[0]
            if evidence.get("role") != "evidence":
                raise SystemExit(f"{candidate}: corpus.segments[0].role {evidence.get('role')!r} != 'evidence'")
            return {
                "path": str(candidate), "sha256": sha256_file(candidate), "partial": candidate.name.endswith(".partial.json"),
                "corpus_sha256": str(corpus["corpus_sha256"]),
                "evidence_segment": {"path": evidence.get("path"), "role": evidence.get("role"),
                                     "input_ids_sha256": evidence.get("input_ids_sha256"),
                                     "rows": evidence.get("rows"), "sequence_length": evidence.get("sequence_length")},
                "segments": len(segments),
                "searched": [str(c) for c in candidates],
            }
    return None


def identity_check(name: str, left: Any, right: Any, *, left_from: str, right_from: str) -> dict[str, Any]:
    return {"check": name, "left": {"from": left_from, "sha256": left}, "right": {"from": right_from, "sha256": right},
            "match": bool(left) and bool(right) and str(left) == str(right)}


def verify_corpus_identity(boundary: dict[str, Any], k3_pass: dict[str, Any] | None,
                           replayed: dict[str, Any] | None, *, allow_unverified: bool) -> dict[str, Any]:
    """Close the manifest <-> boundary map <-> taps and K3-pass <-> boundary map loops.  Every
    compared pair is recorded; any mismatch refuses.  The per-file held_rows metadata check
    runs where the files are loaded (``check_k3_rows_identity``)."""

    checks: list[dict[str, Any]] = [{
        "check": "boundary_map_sha256 seal recomputed", "match": True,
        "left": {"from": "boundary_map.boundary_map_sha256", "sha256": boundary["boundary_map_sha256"]},
        "right": {"from": "estate canonical sha256 over the body minus the seal", "sha256": boundary["boundary_map_sha256_recomputed"]},
    }]
    skipped: list[str] = []
    if k3_pass is not None:
        checks.append(identity_check(
            "k3_pass_receipt evidence ids == boundary_map ids",
            k3_pass["evidence_segment"].get("input_ids_sha256"), boundary_ids_int64_sha256(boundary),
            left_from="k3_dense_pass_receipt.corpus.segments[0].input_ids_sha256 (sha256 over int64 ids bytes)",
            right_from=f"boundary_map.ids_content_sha256[{boundary['file']}].int64",
        ))
    elif allow_unverified:
        skipped.append("no k3_dense_pass receipt: K3 rows NOT tied to the boundary map's ids (--allow-unverified-k3-corpus)")
    else:
        raise SystemExit("no k3_dense_pass receipt found beside --k3-resid or under the default receipt dir; pass "
                         "--k3-pass-receipt (or --allow-unverified-k3-corpus to record the K3 rows as untied to the "
                         "boundary map's ids) -- refusing to pair rows from an unidentified corpus")
    if replayed is not None:
        recorded = replayed["taps_receipt"].get("manifest_sha256")
        boundary_manifest = boundary.get("evidence_manifest_file_sha256")
        if not boundary_manifest:
            raise SystemExit(f"{boundary['path']}: no evidence_manifest_file_sha256; cannot tie the taps' manifest to the "
                             "boundary map -- refusing")
        checks.append(identity_check(
            "taps evidence manifest == boundary_map evidence manifest",
            recorded, boundary_manifest,
            left_from="taps_receipt.evidence.manifest_sha256 (file sha256, verified against the manifest on disk)",
            right_from="boundary_map.evidence_manifest_file_sha256",
        ))
    else:
        skipped.append("taps manifest <-> boundary map: not applicable (trunk positions carried in-file, no taps receipt)")
    failed = [c for c in checks if not c["match"]]
    if failed:
        lines = "; ".join(f"{c['check']}: {c['left']['sha256']!s:.16} ({c['left']['from']}) != {c['right']['sha256']!s:.16} "
                          f"({c['right']['from']})" for c in failed)
        raise SystemExit(f"corpus identity mismatch -- {lines}; refusing to pair rows from different corpora")
    return {
        "verified": k3_pass is not None,
        "k3_pass_receipt": ({k: v for k, v in k3_pass.items() if k != "searched"} if k3_pass else None),
        "k3_pass_receipt_searched": (k3_pass or {}).get("searched"),
        "checks": checks,
        "skipped": skipped,
    }


def check_k3_rows_identity(k3: dict[str, Any], k3_pass: dict[str, Any] | None, *, allow_unverified: bool) -> dict[str, Any]:
    """held_rows.safetensors metadata ``input_ids_sha256`` (k3_dense_pass stamps its corpus sha)
    must equal the pass receipt's ``corpus.corpus_sha256``."""

    stamped = k3["metadata"].get("input_ids_sha256")
    if k3_pass is None:
        if not allow_unverified:
            raise SystemExit(f"{k3['path']}: no k3_dense_pass receipt to check input_ids_sha256 {str(stamped)[:16]} against")
        return {"input_ids_sha256": stamped, "matches_pass_receipt": None}
    if not stamped:
        raise SystemExit(f"{k3['path']}: no input_ids_sha256 metadata; not a k3_dense_pass held_rows file -- refusing")
    if str(stamped) != k3_pass["corpus_sha256"]:
        raise SystemExit(f"{k3['path']}: input_ids_sha256 {str(stamped)[:16]} != k3_dense_pass receipt corpus_sha256 "
                         f"{k3_pass['corpus_sha256'][:16]} ({k3_pass['path']}); the rows come from another corpus "
                         "build -- refusing")
    return {"input_ids_sha256": stamped, "matches_pass_receipt": True}


def flat_pairs(boundary: dict[str, Any], k3_seq_len: int, trunk_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (trunk_flat, k3_flat) pair of the boundary rows, as two int64 tensors."""

    trunk_flat: list[int] = []
    k3_flat: list[int] = []
    for row in boundary["rows"]:
        r = int(row["row_index"])
        for q, k in row["pairs"]:
            q, k = int(q), int(k)
            if not (0 <= q < trunk_seq_len) or not (0 <= k < k3_seq_len):
                raise SystemExit(
                    f"boundary row {r}: pair ({q}, {k}) outside sequence lengths "
                    f"(trunk {trunk_seq_len}, k3 {k3_seq_len})"
                )
            trunk_flat.append(r * trunk_seq_len + q)
            k3_flat.append(r * k3_seq_len + k)
    if not trunk_flat:
        raise SystemExit("boundary map carries no pairs")
    return torch.tensor(trunk_flat, dtype=torch.int64), torch.tensor(k3_flat, dtype=torch.int64)


def position_lookup(positions: torch.Tensor) -> dict[int, int]:
    return {int(p): i for i, p in enumerate(positions.tolist())}


def pair_rows(
    trunk_flat: torch.Tensor,
    k3_flat: torch.Tensor,
    trunk_positions: torch.Tensor,
    k3_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Indices into the trunk / K3 row tensors for every boundary pair present on both sides."""

    trunk_index = position_lookup(trunk_positions)
    k3_index = position_lookup(k3_positions)
    t_rows: list[int] = []
    k_rows: list[int] = []
    t_tokens: list[int] = []
    for t, k in zip(trunk_flat.tolist(), k3_flat.tolist()):
        ti = trunk_index.get(t)
        ki = k3_index.get(k)
        if ti is None or ki is None:
            continue
        t_rows.append(ti)
        k_rows.append(ki)
        t_tokens.append(t)
    return {
        "trunk_rows": torch.tensor(t_rows, dtype=torch.int64),
        "k3_rows": torch.tensor(k_rows, dtype=torch.int64),
        "trunk_tokens": torch.tensor(t_tokens, dtype=torch.int64),
    }


def split_by_trunk_token(trunk_tokens: torch.Tensor, holdout_fraction: float, seed: int) -> dict[str, Any]:
    """Held-out split drawn on unique trunk tokens: disjoint by construction, even when a
    trunk row is paired against several K3 layers."""

    if not 0.0 <= holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be in [0, 1)")
    unique = torch.unique(trunk_tokens)
    count = int(unique.numel())
    generator = torch.Generator().manual_seed(int(seed))
    order = unique[torch.randperm(count, generator=generator)]
    held_count = int(round(holdout_fraction * count))
    if holdout_fraction > 0.0 and held_count == 0 and count > 1:
        held_count = 1
    if held_count >= count:
        raise SystemExit(f"holdout split would consume every one of {count} trunk tokens")
    held_tokens = set(order[:held_count].tolist())
    held_mask = torch.tensor([int(t) in held_tokens for t in trunk_tokens.tolist()], dtype=torch.bool)
    fit_mask = ~held_mask
    if int(fit_mask.sum()) == 0:
        raise SystemExit("holdout split left no fit rows")
    return {
        "fit_mask": fit_mask,
        "held_mask": held_mask,
        "unique_trunk_tokens": count,
        "held_trunk_tokens": held_count,
        "fit_trunk_tokens": count - held_count,
        "split_by": "trunk token",
        "seed": int(seed),
        "holdout_fraction": float(holdout_fraction),
    }


def gate_quantity(unique_trunk_tokens: int, per_k3: Sequence[dict[str, Any]], pooled: int, min_paired_rows: int) -> dict[str, Any]:
    """The 4,096 gate runs on ``min(unique trunk tokens, min paired rows over the K3 layers)``,
    never on the pooled row count (a two-K3-layer pool doubles the rows without adding one
    distinct trunk input: the C_out design would carry duplicate inputs with two targets)."""

    per_layer_min = min((int(e["paired_rows"]) for e in per_k3), default=0)
    rows = min(int(unique_trunk_tokens), per_layer_min)
    tripped_by = "unique_trunk_tokens" if int(unique_trunk_tokens) <= per_layer_min else "min_paired_rows_per_k3_layer"
    return {
        "rows": rows, "quantity": tripped_by, "unique_trunk_tokens": int(unique_trunk_tokens),
        "min_paired_rows_per_k3_layer": per_layer_min, "pooled_paired_rows": int(pooled),
        "min_paired_rows": int(min_paired_rows), "underdetermined": rows < int(min_paired_rows),
        "rule": "min(unique trunk tokens, min paired rows over the pool's K3 layers) >= --min-paired-rows",
    }


# --------------------------------------------------------------------------------------
# Gram diagnostics (section 3b: defects are named)
# --------------------------------------------------------------------------------------


def gram_diagnostics(rows: torch.Tensor, label: str, *, eig: bool = True) -> dict[str, Any]:
    matrix = rows.to(torch.float64)
    if not bool(torch.isfinite(matrix).all()):
        bad = int((~torch.isfinite(matrix)).any(dim=1).sum())
        raise SystemExit(f"{label}: {bad} non-finite rows -> non-finite Gram; refusing (section 3b)")
    variance = matrix.var(dim=0, unbiased=False) if matrix.shape[0] > 1 else matrix.abs().sum(dim=0)
    dead = torch.nonzero(variance <= 0.0).flatten().tolist()
    out: dict[str, Any] = {
        "rows": int(matrix.shape[0]),
        "width": int(matrix.shape[1]),
        "dead_columns": len(dead),
        "dead_column_index_head": dead[:16],
        "row_norm_mean": float(matrix.norm(dim=1).mean()),
        "row_mean_norm": float(matrix.mean(dim=0).norm()) if matrix.shape[0] else 0.0,
        "finite": True,
    }
    if eig:
        gram = matrix.transpose(0, 1) @ matrix
        eigen = torch.linalg.eigvalsh(gram)
        out["gram_eig_min"] = float(eigen.min())
        out["gram_eig_max"] = float(eigen.max())
        out["gram_condition"] = float(eigen.max() / max(float(eigen.min()), 1.0e-300))
        out["gram_rank_rel1e-12"] = int((eigen > 1.0e-12 * float(eigen.max())).sum())
    return out


# --------------------------------------------------------------------------------------
# chart fitting
# --------------------------------------------------------------------------------------


def _regime(rows: int, design_width: int) -> str:
    return "underdetermined" if int(rows) < int(design_width) else "overdetermined"


def _ridge_chart(x_fit: torch.Tensor, y_fit: torch.Tensor, *, seed: int, ladder: bool, fit_bias: bool,
                 ladder_rows: int = 0) -> Any:
    """Lambda chosen on an inner split of the FIT rows, then refitted on every fit row.

    ``ladder_rows`` > 0 walks the ladder on a seeded subsample of that many fit rows; 0 (the
    default) walks it on every fit row.  At 7,168 wide the full ladder is 9 rungs x (one
    ~12k x 7,169^2 fp64 Gram + one Cholesky) -- well under two minutes per direction -- so
    there is no reason to choose the rung in a different regime than the refit: a 4,096-row
    subsample leaves 3,277 inner-fit rows against 7,169 design columns and every rung is
    chosen by how well a strongly under-determined solve interpolates.  Both regimes are
    recorded (``ladder_regime`` / ``refit_regime``); the final chart is ALWAYS refitted on
    every fit row at the chosen rung.
    """

    rows = int(x_fit.shape[0])
    design_width = int(x_fit.shape[1]) + (1 if fit_bias else 0)
    ladder_index = None
    ladder_inner_fit_rows = None
    if ladder and rows >= 5:
        if ladder_rows and rows > int(ladder_rows):
            generator = torch.Generator().manual_seed(int(seed) + 7919)
            ladder_index = torch.randperm(rows, generator=generator)[: int(ladder_rows)].sort().values
            x_l, y_l = x_fit[ladder_index], y_fit[ladder_index]
        else:
            x_l, y_l = x_fit, y_fit
        picked = linalg.fit_ridge_lambda_ladder(x_l, y_l, fit_bias=fit_bias, holdout_fraction=0.2, seed=seed)
        rung = float(picked.notes["ridge_rel"])
        trail = picked.notes["lambda_ladder"]
        ladder_inner_fit_rows = int(picked.notes["fit_rows"])
    else:
        rung = float(linalg.RIDGE_REL)
        trail = None
    refit = linalg.fit_ridge_linear(x_fit, y_fit, ridge_rel=rung, fit_bias=fit_bias, holdout_fraction=0.0, seed=seed)
    refit.notes["lambda_ladder"] = trail
    refit.notes["ladder_rows"] = int(ladder_index.numel()) if ladder_index is not None else (rows if trail else 0)
    refit.notes["ladder_inner_fit_rows"] = ladder_inner_fit_rows
    refit.notes["ridge_rel_selected"] = rung
    refit.notes["design_width"] = design_width
    refit.notes["ladder_regime"] = _regime(ladder_inner_fit_rows, design_width) if ladder_inner_fit_rows else None
    refit.notes["refit_fit_rows"] = rows
    refit.notes["refit_regime"] = _regime(rows, design_width)
    return refit


def _scores(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return linalg.write_scores(prediction, target)


def _apply(rows: torch.Tensor, c: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return rows @ c + bias


def _procrustes_chart(x_fit: torch.Tensor, y_fit: torch.Tensor,
                      x_chart: tuple[torch.Tensor, dict[str, Any]] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Orthonormal C_in [d_x, d_y] on CENTRED rows: plain Procrustes at equal width, otherwise
    Procrustes in the wider side's SVD chart truncated to the narrower width (``x_chart`` =
    the svd arm's chart of the K3 fit rows, shared so the tall SVD is taken once per layer;
    ``svd_chart`` centres internally, so the chart of the raw rows IS the chart of the
    centred rows)."""

    d_x, d_y = int(x_fit.shape[1]), int(y_fit.shape[1])
    if d_x == d_y:
        weight, notes = linalg.orthogonal_procrustes(x_fit, y_fit)  # prediction = x @ weight.T
        c_in = weight.transpose(0, 1).contiguous()
        notes = {**notes, "mode": "equal-width orthogonal Procrustes"}
    elif d_x > d_y:
        chart, chart_notes = x_chart if x_chart is not None else linalg.svd_chart(x_fit, d_y)  # [d_y, d_x]
        weight, notes = linalg.orthogonal_procrustes(x_fit @ chart.transpose(0, 1), y_fit)
        c_in = (chart.transpose(0, 1) @ weight.transpose(0, 1)).contiguous()  # [d_x, d_y]
        notes = {**notes, "mode": "Procrustes in the K3-side SVD chart", "svd_chart": chart_notes}
    else:
        chart, chart_notes = linalg.svd_chart(y_fit, d_x)  # [d_x, d_y]
        weight, notes = linalg.orthogonal_procrustes(x_fit, y_fit @ chart.transpose(0, 1))
        c_in = (weight.transpose(0, 1) @ chart).contiguous()  # [d_x, d_y]
        notes = {**notes, "mode": "Procrustes in the trunk-side SVD chart", "svd_chart": chart_notes}
    notes.pop("scores", None)
    return c_in, notes


def fit_layer_charts(
    x: torch.Tensor,
    y: torch.Tensor,
    fit_mask: torch.Tensor,
    held_mask: torch.Tensor,
    *,
    arms: Sequence[str] = ARMS,
    seed: int = 0,
    ladder: bool = True,
    fit_bias: bool = True,
    ladder_rows: int = 0,
) -> dict[str, Any]:
    """Fit every arm on the fit rows; score forward / backward / round-trip on the held rows.

    ``x`` = K3 rows [n, d_k3], ``y`` = trunk rows [n, d_trunk], paired row-for-row.  Returns
    ``{arm: {"C_in", "C_in_bias", "C_out", "C_out_bias", "held", "fit", "notes"}}``.  The svd
    and procrustes arms are CENTRED: the fit-row means are folded into the biases.
    """

    x = x.to(torch.float64)
    y = y.to(torch.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be paired row-for-row")
    if bool((fit_mask & held_mask).any()):
        raise ValueError("fit and held masks overlap")
    x_fit, y_fit = x[fit_mask], y[fit_mask]
    x_held, y_held = x[held_mask], y[held_mask]
    d_x, d_y = int(x.shape[1]), int(y.shape[1])
    mean_x = x_fit.mean(dim=0)
    mean_y = y_fit.mean(dim=0)
    x_centred, y_centred = x_fit - mean_x, y_fit - mean_y
    centring = {"centred": True, "mean_x_norm": float(mean_x.norm()), "mean_y_norm": float(mean_y.norm()),
                "rule": "fit-row means folded into the biases: y_hat = (x - mean_x) C_in + mean_y; "
                        "x_hat = (y - mean_y) C_out + mean_x"}
    result: dict[str, Any] = {}
    x_chart: tuple[torch.Tensor, dict[str, Any]] | None = None  # K3-side SVD chart, taken once
    for arm in arms:
        started = time.time()
        notes: dict[str, Any] = {}
        if arm == "ridge":
            fwd = _ridge_chart(x_fit, y_fit, seed=seed, ladder=ladder, fit_bias=fit_bias, ladder_rows=ladder_rows)
            bwd = _ridge_chart(y_fit, x_fit, seed=seed, ladder=ladder, fit_bias=fit_bias, ladder_rows=ladder_rows)
            c_in, b_in = fwd.weight.transpose(0, 1).contiguous(), fwd.bias.clone()
            c_out, b_out = bwd.weight.transpose(0, 1).contiguous(), bwd.bias.clone()
            notes = {"in": fwd.notes, "out": bwd.notes,
                     "ridge_lambda_in": fwd.ridge_lambda, "ridge_lambda_out": bwd.ridge_lambda,
                     "regime": {"forward": fwd.notes["refit_regime"], "backward": bwd.notes["refit_regime"],
                                "ladder_forward": fwd.notes["ladder_regime"], "ladder_backward": bwd.notes["ladder_regime"]}}
        elif arm == "svd":
            rank = min(d_x, d_y)
            chart, chart_notes = linalg.svd_chart(x_fit, rank)  # [rank, d_x]; svd_chart centres the rows
            if rank == d_y:
                x_chart = (chart, chart_notes)
            if rank < d_y:  # widen with zeros so C_in is [d_x, d_y]
                pad = torch.zeros(d_y - rank, d_x, dtype=torch.float64)
                chart = torch.cat((chart, pad), dim=0)
            c_in = chart.transpose(0, 1).contiguous()
            c_out = chart.contiguous()
            b_in = (-(mean_x @ c_in)).contiguous()
            b_out = mean_x.clone()
            notes = {**chart_notes, **centring,
                     "control": "orthonormal chart of the centred K3 fit rows; C_out = C_in^T; C_in_bias = -mean_x C_in, "
                                "C_out_bias = mean_x so the round-trip keeps the mean component; the forward score is "
                                "a control, only the round-trip is meaningful"}
        elif arm == "procrustes":
            c_in, notes = _procrustes_chart(x_centred, y_centred, x_chart)
            c_out = c_in.transpose(0, 1).contiguous()
            b_in = (mean_y - mean_x @ c_in).contiguous()
            b_out = (mean_x - mean_y @ c_out).contiguous()
            notes = {**notes, **centring}
        else:
            raise ValueError(f"unknown arm {arm!r}")
        entry: dict[str, Any] = {
            "C_in": c_in, "C_in_bias": b_in, "C_out": c_out, "C_out_bias": b_out, "notes": notes,
            "fit": {
                "forward": _scores(_apply(x_fit, c_in, b_in), y_fit),
                "backward": _scores(_apply(y_fit, c_out, b_out), x_fit),
                "roundtrip": _scores(_apply(_apply(x_fit, c_in, b_in), c_out, b_out), x_fit),
            },
            "seconds": time.time() - started,
        }
        if int(held_mask.sum()) > 0:
            entry["held"] = {
                "forward": _scores(_apply(x_held, c_in, b_in), y_held),
                "backward": _scores(_apply(y_held, c_out, b_out), x_held),
                "roundtrip": _scores(_apply(_apply(x_held, c_in, b_in), c_out, b_out), x_held),
            }
        else:
            entry["held"] = None
        result[arm] = entry
    return result


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k3-resid", required=True, help="k3_dense_pass resid/ dir (L{l:02d}/held_rows.safetensors)")
    ap.add_argument("--trunk-taps", required=True, help="trunk tap dir (L{k:02d}/held_rows.safetensors)")
    ap.add_argument("--boundary-map", required=True, help="S1 boundary_map.json")
    ap.add_argument("--out", required=True, help="charts/ output dir (large: /mnt/lexar/omni-campaign/composite/charts)")
    ap.add_argument("--k3-pass-receipt", default=None,
                    help="k3_dense_pass_receipt.json of the pass that wrote --k3-resid (default: searched beside "
                         f"--k3-resid and under {DEFAULT_K3_PASS_RECEIPT_DIR})")
    ap.add_argument("--allow-unverified-k3-corpus", action="store_true",
                    help="proceed without a k3_dense_pass receipt; the receipt records the K3 rows as NOT tied to the "
                         "boundary map's ids")
    ap.add_argument("--alignment", default=DEFAULT_ALIGNMENT, help="<k3_moe_layers>:<trunk_layers>, default 92:64")
    ap.add_argument("--dense-layer0", default="trunk0", choices=DENSE_LAYER0_POLICIES)
    ap.add_argument("--k3-stream", default="folded", choices=sorted(K3_STREAM_KEYS))
    ap.add_argument("--k3-rows-key", default=None, help="override the K3 rows tensor name")
    ap.add_argument("--trunk-rows-key", default=None, help="override the trunk rows tensor name (e.g. normed)")
    ap.add_argument("--trunk-rows-file", default=None,
                    help=f"trunk layer file name (default: first of {list(TRUNK_FILE_NAMES)} present)")
    ap.add_argument("--evidence-manifest", default=None,
                    help="qwen-fold evidence manifest the taps were drawn from (default: the path the taps "
                         "receipt names); needed only for positions-free tap dumps")
    ap.add_argument("--tap-micro-batch", type=int, default=RUNNER_MICRO_BATCH,
                    help="micro-batch the tap run used (StreamedChimera.run default 8); only matters when the "
                         "tapped rows have unequal lengths")
    ap.add_argument("--boundary-file", default="k3_input_ids", help="boundary map row file to pair (evidence)")
    ap.add_argument("--k3-seq-len", type=int, default=None, help="K3 ids sequence length (default: boundary map)")
    ap.add_argument("--trunk-seq-len", type=int, default=None, help="trunk ids sequence length (default: boundary map)")
    ap.add_argument("--layers", default=None, help="trunk layers to fit, e.g. 0-3,10 (default all)")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list of ridge,svd,procrustes")
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ladder", dest="ladder", action="store_false", default=True,
                    help="skip the lambda ladder and use RIDGE_REL")
    ap.add_argument("--no-bias", dest="fit_bias", action="store_false", default=True)
    ap.add_argument("--ladder-rows", type=int, default=0,
                    help="fit rows sampled for the lambda ladder (default 0 = every fit row, so the rung is chosen "
                         "in the refit's regime; a subsample below the design width is recorded as "
                         "ladder_regime=underdetermined); the chart is always refitted on every fit row")
    ap.add_argument("--min-paired-rows", type=int, default=DEFAULT_MIN_PAIRED_ROWS,
                    help="gate on min(unique trunk tokens, min paired rows per K3 layer), not the pooled count")
    ap.add_argument("--allow-underdetermined", action="store_true",
                    help="fit a layer whose gate quantity is below --min-paired-rows and flag it")
    ap.add_argument("--max-rows", type=int, default=0, help="cap paired rows per trunk layer (0 = all)")
    ap.add_argument("--chart-dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--skip-control-charts", action="store_true",
                    help="fit + score the svd / procrustes arms but do not write their chart files (numbers stay "
                         "in the receipt); saves 2/3 of the footprint")
    ap.add_argument("--allow-root-disk", action="store_true",
                    help=f"allow --out under {ROOT_DISK_PREFIX} (the root disk); large artefacts belong under "
                         "/mnt/lexar/omni-campaign/<lane>/")
    ap.add_argument("--refit", action="store_true",
                    help="refit requested layers even when their L{k}/layer.json + arm files verify by sha256")
    ap.add_argument("--no-gram-eig", dest="gram_eig", action="store_false", default=True,
                    help="skip the Gram eigen diagnostics (large widths)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave)")
    return ap


def parse_layers(text: str | None, count: int) -> list[int]:
    if not text:
        return list(range(count))
    picked: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-")
            picked.extend(range(int(a), int(b) + 1))
        else:
            picked.append(int(piece))
    bad = [k for k in picked if not 0 <= k < count]
    if bad:
        raise SystemExit(f"--layers {text!r}: trunk layers {bad} outside 0..{count - 1}")
    return sorted(set(picked))


def chart_state(arm: dict[str, Any], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {name: arm[name].to(dtype).contiguous() for name in ("C_in", "C_in_bias", "C_out", "C_out_bias")}


def held_summary(arm_entry: dict[str, Any]) -> dict[str, Any] | None:
    """Held-out cos / relL2 per direction, or None when there are no held rows -- fit scores
    are never relabelled as held (they stay under ``fit``)."""

    scored = arm_entry.get("held")
    if not scored:
        return None
    return {
        direction: {"cos": scored[direction]["cos"], "rel_l2": scored[direction]["rel_l2"],
                    "cos_p05": scored[direction]["cos_p05"], "rows": scored[direction]["rows"]}
        for direction in ("forward", "backward", "roundtrip")
    }


def fit_signature(args: argparse.Namespace, boundary: dict[str, Any], k3_key: str, arms: Sequence[str]) -> dict[str, Any]:
    """What a layer's charts depend on besides the row files (whose sha256s the record carries)."""

    return {
        "tool_version": TOOL_VERSION, "arms": list(arms), "k3_stream": args.k3_stream, "k3_rows_key": k3_key,
        "trunk_rows_key": args.trunk_rows_key, "holdout_fraction": float(args.holdout_fraction), "seed": int(args.seed),
        "ladder": bool(args.ladder), "ladder_rows": int(args.ladder_rows), "fit_bias": bool(args.fit_bias),
        "chart_dtype": args.chart_dtype, "alignment": args.alignment, "dense_layer0": args.dense_layer0,
        "max_rows": int(args.max_rows), "boundary_map_sha256": boundary["boundary_map_sha256"],
        "boundary_file": boundary["file"], "skip_control_charts": bool(args.skip_control_charts),
    }


def layer_dir_status(layer_dir: Path, signature: dict[str, Any] | None, input_paths: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate an existing ``L{k}/layer.json``: every recorded arm file present with its sha256,
    the fit signature equal to ``signature`` (when given) and the input row files unchanged
    (``input_paths`` = {"trunk": path, "k3": {layer: path}} to hash now, when given).  None when
    no layer.json exists; otherwise ``{"record", "valid", "reasons", "layer_json_sha256"}``."""

    layer_json = layer_dir / "layer.json"
    if not layer_json.is_file():
        return None
    reasons: list[str] = []
    try:
        record = json.loads(layer_json.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        return {"record": None, "valid": False, "reasons": [f"layer.json unreadable: {type(error).__name__}"],
                "layer_json_sha256": sha256_file(layer_json)}
    if not isinstance(record, dict) or "trunk_layer" not in record or "files" not in record:
        reasons.append("layer.json lacks trunk_layer / files")
    else:
        for arm, written in (record.get("files") or {}).items():
            if written is None:
                continue
            path = Path(written.get("path", ""))
            if not path.is_absolute():
                path = layer_dir / path.name
            if not path.is_file():
                reasons.append(f"{arm}: {path} missing")
            elif sha256_file(path) != written.get("file_sha256"):
                reasons.append(f"{arm}: {path} sha256 differs from layer.json")
        if signature is not None and record.get("fit_signature") != signature:
            reasons.append("fit signature differs (args / tool version / boundary map)")
        if input_paths is not None:
            trunk_sha = (record.get("trunk_taps") or {}).get("sha256")
            if not input_paths.get("trunk") or sha256_file(input_paths["trunk"]) != trunk_sha:
                reasons.append("trunk rows file changed since the layer was fitted")
            recorded_k3 = {int(e["k3_layer"]): e.get("sha256") for e in record.get("k3_rows") or []}
            for l, path in (input_paths.get("k3") or {}).items():
                if int(l) not in recorded_k3 or not os.path.isfile(path) or sha256_file(path) != recorded_k3[int(l)]:
                    reasons.append(f"K3 layer {l} rows file changed since the layer was fitted")
            if set(recorded_k3) != {int(l) for l in (input_paths.get("k3") or {})}:
                reasons.append("K3 layer pool differs from the alignment")
    return {"record": record, "valid": not reasons, "reasons": reasons, "layer_json_sha256": sha256_file(layer_json)}


def scan_layer_dirs(out_dir: Path) -> list[int]:
    found: list[int] = []
    for entry in sorted(out_dir.iterdir()) if out_dir.is_dir() else []:
        if entry.is_dir() and entry.name.startswith("L") and entry.name[1:].isdigit() and (entry / "layer.json").is_file():
            found.append(int(entry.name[1:]))
    return sorted(set(found))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.threads > 0:
        torch.set_num_threads(int(args.threads))
    moe_layers, trunk_layers = parse_alignment(args.alignment)
    table = alignment_table(moe_layers, trunk_layers, args.dense_layer0)
    summary = alignment_summary(table, moe_layers, args.dense_layer0)
    if not summary["covered_exactly_once"]:
        raise SystemExit(f"alignment does not cover every K3 layer exactly once: {summary}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"--arms: unknown arm {arm!r} (choose from {ARMS})")
    arms_written = [a for a in arms if not (args.skip_control_charts and a in CONTROL_ARMS)]
    layers = parse_layers(args.layers, trunk_layers)

    if under_root_disk(args.out) and not args.allow_root_disk:
        raise SystemExit(f"--out {args.out} resolves under {ROOT_DISK_PREFIX} (the root disk); three fp32 arms at "
                         f"7,168 x 5,120 are 56.4 GB over 64 layers -- write charts under /mnt/lexar/omni-campaign/"
                         f"<lane>/ or pass --allow-root-disk")

    boundary = load_boundary_map(args.boundary_map, args.boundary_file)
    k3_seq_len = int(args.k3_seq_len or boundary["sequence_length"])
    trunk_seq_len = int(args.trunk_seq_len or boundary["sequence_length"])
    trunk_flat, k3_flat = flat_pairs(boundary, k3_seq_len, trunk_seq_len)
    k3_key = args.k3_rows_key or K3_STREAM_KEYS[args.k3_stream]
    trunk_keys = (args.trunk_rows_key,) if args.trunk_rows_key else TRUNK_ROW_KEYS
    chart_dtype = torch.float32 if args.chart_dtype == "float32" else torch.float64
    k3_pass = load_k3_pass_receipt(args.k3_pass_receipt, args.k3_resid)
    if args.k3_pass_receipt and k3_pass is None:
        raise SystemExit(f"--k3-pass-receipt {args.k3_pass_receipt}: not a file")

    out_dir = Path(args.out)
    previous_receipt_sha = sha256_file(out_dir / "receipt.json") if (out_dir / "receipt.json").is_file() else None
    out_dir.mkdir(parents=True, exist_ok=True)
    alignment_doc = {"schema": "chimera_omni.composite.c17_alignment.v1", "summary": summary, "table": table}
    (out_dir / "alignment.json").write_text(json.dumps(alignment_doc, indent=2) + "\n", encoding="utf-8")

    # positions-free tap dumps (the runner's layout): replay the selection once, reuse per layer
    taps = load_taps_receipt(args.trunk_taps)
    replayed: dict[str, Any] | None = None
    fallback_positions: torch.Tensor | None = None
    if taps is not None:
        replayed = replayed_trunk_positions(args.trunk_taps, taps, args.evidence_manifest,
                                            micro_batch=int(args.tap_micro_batch), sequence_length=trunk_seq_len)
        fallback_positions = replayed["positions"]
        print(f"trunk taps: replayed {int(fallback_positions.numel())} positions from {taps['path']} "
              f"(stride {replayed['replay']['stride']}, rows {replayed['replay']['tapped_batch_indices']})", flush=True)
    identity = verify_corpus_identity(boundary, k3_pass, replayed, allow_unverified=bool(args.allow_unverified_k3_corpus))
    for check in identity["checks"]:
        print(f"identity: {check['check']}: {str(check['left']['sha256'])[:16]} == {str(check['right']['sha256'])[:16]}", flush=True)
    for note in identity["skipped"]:
        print(f"identity: SKIPPED {note}", flush=True)

    signature = fit_signature(args, boundary, k3_key, arms)
    projected: dict[str, Any] | None = None
    fitted_records: list[dict[str, Any]] = []
    reused_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    for k in layers:
        row = table[k]
        layer_dir = out_dir / f"L{k:02d}"
        layer_started = time.time()
        trunk_path = resolve_trunk_file(args.trunk_taps, k, args.trunk_rows_file)
        k3_paths = {l: os.path.join(args.k3_resid, f"L{l:02d}", "held_rows.safetensors") for l in row["k3_layers"]}
        previous = layer_dir_status(layer_dir, signature, {"trunk": trunk_path, "k3": k3_paths})
        if previous is not None and previous["valid"] and not args.refit:
            record = previous["record"]
            record["status"] = "reused"
            record["reuse"] = {"reason": "layer.json + arm files verified by sha256; inputs and fit signature unchanged",
                               "layer_json_sha256": previous["layer_json_sha256"]}
            record["layer_json_sha256"] = previous["layer_json_sha256"]
            reused_records.append(record)
            print(f"L{k:02d} <- K3 {row['k3_layers']} REUSED (verified by sha256; --refit to redo)", flush=True)
            continue
        refit_reason = None
        if previous is not None:
            refit_reason = "--refit" if previous["valid"] else previous["reasons"]
        trunk = load_rows_file(trunk_path, trunk_keys, trunk_seq_len, fallback_positions=fallback_positions)
        tap_sha_match = check_tap_file_sha(trunk, taps, k)
        pooled_x: list[torch.Tensor] = []
        pooled_y_index: list[torch.Tensor] = []
        pooled_tokens: list[torch.Tensor] = []
        per_k3: list[dict[str, Any]] = []
        for l in row["k3_layers"]:
            k3 = load_rows_file(k3_paths[l], (k3_key,), k3_seq_len)
            k3_identity = check_k3_rows_identity(k3, k3_pass, allow_unverified=bool(args.allow_unverified_k3_corpus))
            pairs = pair_rows(trunk_flat, k3_flat, trunk["positions"], k3["positions"])
            per_k3.append({
                "k3_layer": l, "path": k3["path"], "sha256": k3["sha256"], "rows_key": k3["rows_key"],
                "source_dtype": k3["dtype"], "rows_available": int(k3["rows"].shape[0]), "width": int(k3["rows"].shape[1]),
                "paired_rows": int(pairs["k3_rows"].numel()),
                "underdetermined": int(pairs["k3_rows"].numel()) < int(args.min_paired_rows),
                **k3_identity,
            })
            pooled_x.append(k3["rows"][pairs["k3_rows"]])
            pooled_y_index.append(pairs["trunk_rows"])
            pooled_tokens.append(pairs["trunk_tokens"])
            del k3
        x = torch.cat(pooled_x, dim=0) if pooled_x else torch.zeros(0, 0, dtype=torch.float64)
        y_index = torch.cat(pooled_y_index, dim=0) if pooled_y_index else torch.zeros(0, dtype=torch.int64)
        tokens = torch.cat(pooled_tokens, dim=0) if pooled_tokens else torch.zeros(0, dtype=torch.int64)
        paired = int(y_index.numel())
        unique_tokens = int(torch.unique(tokens).numel()) if paired else 0
        gate = gate_quantity(unique_tokens, per_k3, paired, int(args.min_paired_rows))
        record: dict[str, Any] = {
            "trunk_layer": k, "k3_layers": row["k3_layers"], "moe_layers": row["moe_layers"],
            "status": "fitted", "refit_reason": refit_reason, "fit_signature": signature,
            "trunk_taps": {"path": trunk["path"], "sha256": trunk["sha256"], "rows_key": trunk["rows_key"],
                           "position_key": trunk["position_key"], "source_dtype": trunk["dtype"],
                           "rows_available": int(trunk["rows"].shape[0]), "width": int(trunk["rows"].shape[1]),
                           "sha256_matches_taps_receipt": tap_sha_match},
            "k3_rows": per_k3,
            "paired_rows": paired,
            "paired_rows_per_k3_layer": {str(e["k3_layer"]): e["paired_rows"] for e in per_k3},
            "unique_trunk_tokens": unique_tokens,
            "min_paired_rows": int(args.min_paired_rows),
            "gate": gate,
            "underdetermined": gate["underdetermined"],
        }
        if record["underdetermined"] and not args.allow_underdetermined:
            message = (f"trunk layer {k}: {gate['rows']} {gate['quantity']} < {args.min_paired_rows} (pooled {paired} paired rows, "
                       f"unique trunk tokens {unique_tokens}, K3 layers {row['k3_layers']}, per layer "
                       f"{record['paired_rows_per_k3_layer']}); refusing an under-determined chart -- "
                       f"pass --allow-underdetermined to fit and flag it")
            print(message, file=sys.stderr)
            raise SystemExit(message)
        if paired < 2:
            raise SystemExit(f"trunk layer {k}: {paired} paired rows; nothing to fit")
        if args.max_rows and paired > args.max_rows:
            generator = torch.Generator().manual_seed(int(args.seed) + k)
            keep = torch.randperm(paired, generator=generator)[: int(args.max_rows)].sort().values
            x, y_index, tokens = x[keep], y_index[keep], tokens[keep]
            record["rows_capped_to"] = int(args.max_rows)
        y = trunk["rows"][y_index]
        del trunk
        split = split_by_trunk_token(tokens, args.holdout_fraction, args.seed)
        fit_mask, held_mask = split.pop("fit_mask"), split.pop("held_mask")
        record["split"] = {**split, "fit_rows": int(fit_mask.sum()), "held_rows": int(held_mask.sum()),
                           "disjoint": not bool((fit_mask & held_mask).any())}
        record["gram"] = {
            "k3_fit": gram_diagnostics(x[fit_mask], f"trunk layer {k} K3 rows", eig=args.gram_eig),
            "trunk_fit": gram_diagnostics(y[fit_mask], f"trunk layer {k} trunk rows", eig=args.gram_eig),
        }
        if projected is None:
            projected = projected_chart_bytes(int(x.shape[1]), int(y.shape[1]), chart_dtype, arms_written, len(layers))
            print(f"projected chart bytes: {projected['bytes_per_layer']:,} per layer ({len(arms_written)} arm files x "
                  f"{projected['bytes_per_arm']:,}) x {len(layers)} layers = {projected['bytes_total']:,} -> {out_dir}", flush=True)
        fitted = fit_layer_charts(x, y, fit_mask, held_mask, arms=arms, seed=args.seed,
                                  ladder=args.ladder, fit_bias=args.fit_bias, ladder_rows=int(args.ladder_rows))
        record["arms"] = {}
        record["files"] = {}
        for arm, entry in fitted.items():
            written = None
            if arm in arms_written:
                written = receipts.write_tensors(chart_state(entry, chart_dtype), layer_dir / ARM_FILES[arm])
            record["files"][arm] = written
            record["arms"][arm] = {
                "held": entry["held"], "fit": entry["fit"], "notes": entry["notes"],
                "seconds": entry["seconds"], "held_summary": held_summary(entry),
                "held_summary_source": "held" if entry["held"] else None,
                "parameters": written["parameters"] if written else None, "chart_dtype": args.chart_dtype,
                "written": written is not None,
                "convention": "y_hat = x_k3 @ C_in + C_in_bias; x_hat = y_trunk @ C_out + C_out_bias",
            }
        if "ridge" in record["arms"]:
            record["ridge_regime"] = record["arms"]["ridge"]["notes"]["regime"]
        record["seconds"] = time.time() - layer_started
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_json = layer_dir / "layer.json"
        layer_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        record["layer_json_sha256"] = sha256_file(layer_json)
        fitted_records.append(record)
        ridge = record["arms"].get("ridge")
        line = ""
        if ridge and ridge["held_summary"]:
            hs = ridge["held_summary"]
            line = (f" ridge held cos in {hs['forward']['cos']:.4f} relL2 {hs['forward']['rel_l2']:.4f} | "
                    f"out cos {hs['backward']['cos']:.4f} relL2 {hs['backward']['rel_l2']:.4f} | "
                    f"rt cos {hs['roundtrip']['cos']:.4f}")
        elif ridge:
            line = " (no held rows: fit scores only)"
        print(f"L{k:02d} <- K3 {row['k3_layers']} paired {paired} (fit {record['split']['fit_rows']} / "
              f"held {record['split']['held_rows']}){' UNDERDETERMINED' if record['underdetermined'] else ''}{line} "
              f"[{record['seconds']:.1f}s]", flush=True)
        del x, y, fitted

    # merge: every L*/layer.json present that this invocation did not touch
    touched = {r["trunk_layer"] for r in fitted_records} | {r["trunk_layer"] for r in reused_records}
    for k in scan_layer_dirs(out_dir):
        if k in touched:
            continue
        previous = layer_dir_status(out_dir / f"L{k:02d}", None, None)
        if previous is None:
            continue
        if previous["valid"]:
            record = previous["record"]
            record["status"] = "reused"
            record["reuse"] = {"reason": "not requested this invocation; layer.json + arm files verified by sha256",
                               "layer_json_sha256": previous["layer_json_sha256"]}
            record["layer_json_sha256"] = previous["layer_json_sha256"]
            reused_records.append(record)
        else:
            invalid_records.append({"trunk_layer": k, "status": "invalid", "reasons": previous["reasons"],
                                    "layer_json_sha256": previous["layer_json_sha256"],
                                    "path": str(out_dir / f"L{k:02d}" / "layer.json")})
            print(f"L{k:02d}: INVALID layer dir excluded from the receipt: {previous['reasons']}", file=sys.stderr, flush=True)
    valid_records = sorted(fitted_records + reused_records, key=lambda r: int(r["trunk_layer"]))
    all_records = sorted(valid_records + invalid_records, key=lambda r: int(r["trunk_layer"]))

    def _stat(direction: str, key: str) -> dict[str, float] | None:
        values = [r["arms"]["ridge"]["held_summary"][direction][key] for r in valid_records
                  if r.get("arms", {}).get("ridge") and r["arms"]["ridge"].get("held_summary")]
        if not values:
            return None
        return {"min": min(values), "max": max(values), "mean": sum(values) / len(values), "layers": len(values)}

    def _regime_layers(direction: str) -> list[int]:
        return [r["trunk_layer"] for r in valid_records
                if (r.get("ridge_regime") or {}).get(direction) == "underdetermined"]

    body = {
        "schema": RECEIPT_SCHEMA,
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "tool_sha256": sha256_file(os.path.abspath(__file__)),
        "source_sha256": receipts.source_sha256([str(MEDIA_ROOT / "mediafold" / "linalg.py"),
                                                 str(MEDIA_ROOT / "mediafold" / "receipts.py")]),
        "mediafold_receipt_schema": receipts.RECEIPT_SCHEMA,
        "media_root": str(MEDIA_ROOT),
        "args": {k: v for k, v in vars(args).items()},
        "alignment": alignment_doc,
        "alignment_json_sha256": sha256_file(out_dir / "alignment.json"),
        "boundary_map": {k: v for k, v in boundary.items() if k != "rows"},
        "boundary_pairs_total": int(trunk_flat.numel()),
        "k3_seq_len": k3_seq_len, "trunk_seq_len": trunk_seq_len,
        "k3_rows_key": k3_key,
        "corpus_identity": identity,
        "trunk_tap_contract": f"L{{k:02d}}/{' | '.join(TRUNK_FILE_NAMES)} with rows under one of {list(trunk_keys)}; "
                              "positions from held_positions (flat row*T+pos into the Qwen evidence ids, row order == "
                              "boundary_map row_index) or positions [rows,2], else replayed from taps_receipt.json + "
                              "the evidence manifest (run_omni_frozen_review selection rule); file sha256 checked "
                              "against taps_receipt.layers[].sha256 when recorded",
        "trunk_positions": ({k: v for k, v in replayed.items() if k != "positions"} if replayed
                            else {"source": "in-file positions"}),
        "projected_bytes": projected,
        "control_charts_written": not bool(args.skip_control_charts),
        "fit_signature": signature,
        "previous_receipt_sha256": previous_receipt_sha,
        "layers_requested": layers,
        "layers_fitted": [r["trunk_layer"] for r in fitted_records],
        "layers_reused": sorted(r["trunk_layer"] for r in reused_records),
        "layers_invalid": [r["trunk_layer"] for r in invalid_records],
        "layers_present": [r["trunk_layer"] for r in valid_records],
        "layers_underdetermined": [r["trunk_layer"] for r in valid_records if r["underdetermined"]],
        "layers_ridge_refit_underdetermined": {"forward": _regime_layers("forward"), "backward": _regime_layers("backward")},
        "layers_ridge_ladder_underdetermined": {"forward": _regime_layers("ladder_forward"), "backward": _regime_layers("ladder_backward")},
        "paired_rows_per_trunk_layer": {str(r["trunk_layer"]): r["paired_rows"] for r in valid_records},
        "gate_rows_per_trunk_layer": {str(r["trunk_layer"]): r["gate"]["rows"] for r in valid_records if r.get("gate")},
        "ridge_held": {
            "forward_cos": _stat("forward", "cos"), "forward_rel_l2": _stat("forward", "rel_l2"),
            "backward_cos": _stat("backward", "cos"), "backward_rel_l2": _stat("backward", "rel_l2"),
            "roundtrip_cos": _stat("roundtrip", "cos"), "roundtrip_rel_l2": _stat("roundtrip", "rel_l2"),
            "source": "held rows only; None when no layer had a holdout",
        },
        "layers": all_records,
        "runtime": receipts.runtime_block(),
        "seconds": time.time() - started,
        "note": "diagnostic numbers only: cos 0.97 with zero functional transfer is the field's warning; the "
                "chart's worth is read in S5's veto / energy numbers (run-book S4). The run book's '>= 4,096 rows "
                "is over-determined' line is false at 7,168 wide: see ridge_regime per layer.",
    }
    receipt_path = out_dir / "receipt.json"
    receipt_sha = receipts.write_receipt(receipt_path, body)
    print(f"receipt {receipt_path} sha256 {receipt_sha} fitted {len(fitted_records)}/{len(layers)} reused "
          f"{len(reused_records)} invalid {len(invalid_records)} underdetermined {len(body['layers_underdetermined'])} "
          f"{body['seconds']:.1f}s", flush=True)
    body["receipt_sha256"] = receipt_sha
    body["receipt_path"] = str(receipt_path)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
