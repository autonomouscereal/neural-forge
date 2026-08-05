#!/usr/bin/env python3
"""Fold a real chimera-omni model at exact 4x parameter compaction.

``chimera-omni`` is the target platform: an encoder-free native omnimodal
decoder whose media path is one shared boundary projection and whose output
side emits continuous media from one shared head.  This runner is the first
end-to-end evidence that the folding platform reaches it.  It:

1. builds a **real** tiny chimera-omni model through chimera-omni's own
   construction APIs -- ``ChimeraConfig`` and ``ChimeraOmniModel``, imported
   from the target repository's ``src`` directory, never vendored -- and trains
   it on chimera-omni's own tiny-proof batches so the parent has something to
   lose;
2. compiles both the parent and a 4x narrower child into the canonical IR with
   :class:`~neural_forge.dynamic_model_inspector.ChimeraOmniCircuitInspector`;
3. proves the compaction is **exactly** 4x by unique learned-parameter count,
   with no quantization, no sparsity, no storage sharing and no sidecar;
4. asks :class:`~neural_forge.dynamic_fold_planner.DynamicFoldPlanner` what it
   can cover and records its refusal verbatim;
5. executes a zero-gradient closed-form fold with the repository's own
   ``hf_closed_form_fit`` kernels, inside the repository's fold-only guard;
6. reads decoded text and *emitted media* out of the folded child, renders the
   media with chimera-omni's own renderers, and scores the child against both
   the trained parent and a blank child of identical size.

Scope, stated up front
----------------------
* The compaction axis is the **residual width** and only the residual width.
  Head geometry (``num_heads * attention_head_dim``) and the feed-forward
  intermediate width are held fixed, which is what makes every parameter block
  exactly linear in ``hidden_size`` and therefore makes ``4x`` an identity
  rather than a rounding.  It also means every fit has matching non-residual
  widths on the other side, so no head-folding or neuron-selection kernel is
  involved.
* The tiny configuration deliberately uses **only** the ``local`` sequence
  mixer and the dense channel mixer.  ``kda``, ``csa``, ``hca``, ``gated_mla``
  and ``stable_latent_moe`` are *avoided*, not covered, and the avoided set is
  written into the report.  The inspector compiles all of them; the fold does
  not attempt them.
* The residual crossing uses **one global orthonormal chart** fitted from the
  parent's own pooled residual rows.  The production path -- a sealed
  :class:`~neural_forge.hf_causal_cross_width_charts.ResidualBoundaryChartSet`
  with one chart per residual-write boundary -- is bound to
  ``HFCausalCircuitEvidence`` and does not accept a non-Hugging-Face parent, so
  it is out of scope here and is named as a gap rather than approximated
  silently.
* Nothing here claims retained intelligence.  What it claims is: the child is
  exactly a quarter of the parent, it was built without a gradient step, it
  runs, it emits media, and it beats a blank child of the same size on the
  parent's own objective by the margins printed in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    CircuitKind,
    canonical_sha256,
    emitted_media_interfaces,
)
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    DynamicFoldExecutionError,
    fingerprint_model_state,
    run_fold_only_callback,
)
from neural_forge.dynamic_fold_planner import (  # noqa: E402
    CAUSAL_LAYER_BRANCH_METHODS,
    DynamicFoldPlanner,
    DynamicFoldPlanningError,
)
from neural_forge.dynamic_model_inspector import (  # noqa: E402
    ChimeraOmniCircuitInspector,
    inspect_model_circuits,
)
from neural_forge.hf_closed_form_fit import (  # noqa: E402
    fit_diagonal_norm,
    fit_ridge_linear,
    normalize_rows,
)


RUNNER_SCHEMA = "neural-forge-chimera-omni-toy-fold-runner-v1"
CHIMERA_OMNI_ENVIRONMENT_KEY = "NEURAL_FORGE_CHIMERA_OMNI_REPO"
DEFAULT_CHIMERA_OMNI_REPO = Path(
    "C:/Users/cereal/Documents/ai training/chimera-omni"
)
CHIMERA_OMNI_SOURCE_FILES = (
    "architecture_manifest.json",
    "src/chimera_omni/config.py",
    "src/chimera_omni/modeling/chimera.py",
    "src/chimera_omni/modeling/layers.py",
    "src/chimera_omni/native_media.py",
    "src/chimera_omni/native_segments.py",
    "src/chimera_omni/proof/tiny_multimodal.py",
    "src/chimera_omni/proof/media_render.py",
)

# The sequence-mixing and channel-mixing families this fold deliberately does
# not attempt.  Recorded in the report so "covered" never reads as "all of it".
AVOIDED_BLOCK_FAMILIES = ("kda", "csa", "hca", "gated_mla")
AVOIDED_CHANNEL_MIXERS = ("stable_latent_moe",)

COMPACTION_FACTOR = 4
PARENT_HIDDEN_SIZE = 128
CHILD_HIDDEN_SIZE = PARENT_HIDDEN_SIZE // COMPACTION_FACTOR

STAGES = (
    "configuration",
    "parent",
    "inspection",
    "budget",
    "plan",
    "fold",
    "behavior",
)


class ChimeraOmniFoldError(RuntimeError):
    """The run cannot proceed and nothing was written past the last seal."""


# ---------------------------------------------------------------------------
# Sealed, content-addressed I/O
# ---------------------------------------------------------------------------


def _canonical_bytes(body: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def seal_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``body`` plus the canonical digest of everything else in it."""

    content = {key: value for key, value in body.items() if key != "receipt_sha256"}
    return {**content, "receipt_sha256": canonical_sha256(content)}


def verify_receipt(document: Mapping[str, Any]) -> bool:
    """Recompute a receipt seal over its own body."""

    content = {key: value for key, value in document.items() if key != "receipt_sha256"}
    return document.get("receipt_sha256") == canonical_sha256(content)


def write_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    sealed = seal_receipt(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(sealed))
    return sealed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path, skip: Sequence[str] = ()) -> list[dict[str, Any]]:
    skipped = set(skip)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name in skipped:
            continue
        rows.append(
            {
                "path": name,
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return rows


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return hashlib.sha256(array.numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# The target repository
# ---------------------------------------------------------------------------


def resolve_chimera_omni_repo(explicit: str | None = None) -> Path:
    """Locate the chimera-omni checkout without copying anything out of it."""

    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    environment = os.environ.get(CHIMERA_OMNI_ENVIRONMENT_KEY, "")
    if environment:
        candidates.append(Path(environment))
    candidates.append(DEFAULT_CHIMERA_OMNI_REPO)
    candidates.append(ROOT.parent / "chimera-omni")
    for candidate in candidates:
        source = candidate / "src" / "chimera_omni"
        if source.is_dir():
            return candidate.resolve()
    raise ChimeraOmniFoldError(
        "Could not locate the chimera-omni repository; pass "
        "--chimera-omni-repo or set "
        f"{CHIMERA_OMNI_ENVIRONMENT_KEY}. Tried: "
        + ", ".join(str(value) for value in candidates)
        + "."
    )


def import_chimera_omni(repo: Path) -> dict[str, Any]:
    """Put the target repo's ``src`` on the path and import its real modules."""

    source = repo / "src"
    if not (source / "chimera_omni" / "__init__.py").is_file():
        raise ChimeraOmniFoldError(f"{source} is not a chimera-omni source tree.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import chimera_omni.config as chimera_config
    import chimera_omni.modeling as chimera_modeling
    import chimera_omni.native_segments as chimera_native_segments
    import chimera_omni.proof.media_render as chimera_media_render
    import chimera_omni.proof.tiny_multimodal as chimera_tiny_proof

    return {
        "config": chimera_config,
        "modeling": chimera_modeling,
        "native_segments": chimera_native_segments,
        "media_render": chimera_media_render,
        "tiny_proof": chimera_tiny_proof,
    }


def chimera_source_identity(repo: Path) -> dict[str, Any]:
    rows = []
    for name in CHIMERA_OMNI_SOURCE_FILES:
        path = repo / name
        if not path.is_file():
            raise ChimeraOmniFoldError(f"chimera-omni is missing {name}.")
        rows.append(
            {
                "path": name,
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return {
        "repository": str(repo),
        "read_only": True,
        "vendored": False,
        "files": rows,
    }


# ---------------------------------------------------------------------------
# Configurations: exact 4x by construction
# ---------------------------------------------------------------------------


def coverable_config(chimera_config: Any, *, hidden_size: int) -> Any:
    """One chimera config whose block families the fold actually covers.

    Every learned tensor of this topology is exactly linear in ``hidden_size``:
    the attention projections span ``num_heads * attention_head_dim`` on their
    non-residual side and the feed-forward spans ``intermediate_size`` on
    theirs, and both are pinned.  Halving twice therefore divides the unique
    parameter count by exactly four with nothing left over.
    """

    config = chimera_config.ChimeraConfig(
        vocab_size=1024,
        hidden_size=int(hidden_size),
        intermediate_size=320,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        max_sequence_length=64,
        block_pattern=("local",),
        residual_mode="block_attnres",
        attnres_block_size=2,
        attention_head_dim=32,
        feed_forward_kind="dense",
        activation_kind="silu",
        native_feature_dim=64,
        native_position_dim=4,
        native_type_count=8,
        mtp_heads=1,
        aux_heads=("flow", "grounding", "sync"),
        local_window=16,
    )
    config.validate()
    return config


def unique_parameter_count(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


# ---------------------------------------------------------------------------
# Batches: chimera-omni's own tiny-proof data
# ---------------------------------------------------------------------------


def build_batch(
    tiny_proof: Any,
    config: Any,
    *,
    batch_size: int,
    seq_len: int,
    max_media_tokens: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, int]]]:
    generator = random.Random(int(seed))
    return tiny_proof.make_batch(
        cfg=config,
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        max_media_tokens=int(max_media_tokens),
        device=torch.device("cpu"),
        rng=generator,
    )


def batch_digest(batch: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256(
        {
            "kind": "chimera_tiny_proof_batch",
            "tensors": [
                [
                    name,
                    [int(size) for size in batch[name].shape],
                    tensor_sha256(batch[name]),
                ]
                for name in sorted(batch)
            ],
        }
    )


def train_parent(
    tiny_proof: Any,
    model: torch.nn.Module,
    config: Any,
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    max_media_tokens: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Train the parent with chimera-omni's own tiny-proof objective.

    The loop mirrors ``chimera_omni.proof.tiny_multimodal.train_tiny_proof``;
    the only difference is the block pattern, because that function pins the
    ``kda/local/csa/hca`` pattern this fold deliberately avoids.  The pass
    criterion is chimera-omni's own: the mean of the last ten losses must be
    below the mean of the first ten.
    """

    generator = random.Random(int(seed))
    torch.manual_seed(int(seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=0.01
    )
    model.train()
    losses: list[float] = []
    started = time.perf_counter()
    for _step in range(int(steps)):
        batch, _specs = tiny_proof.make_batch(
            cfg=config,
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            max_media_tokens=int(max_media_tokens),
            device=torch.device("cpu"),
            rng=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output["loss"]
        if loss is None:
            raise ChimeraOmniFoldError("chimera-omni returned no training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None
    window = max(1, min(10, len(losses)))
    first = sum(losses[:window]) / window
    last = sum(losses[-window:]) / window
    return {
        "objective": "chimera_omni_tiny_multimodal_text_ce_plus_native_media_mse",
        "loop_mirrors": "chimera_omni.proof.tiny_multimodal.train_tiny_proof",
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_first_10_loss": first,
        "mean_last_10_loss": last,
        "loss_drop_ratio": last / max(1.0e-8, first),
        "chimera_tiny_proof_status": "passed" if last < first else "needs_more_steps",
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "losses": [round(value, 6) for value in losses],
    }


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


def _rows(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor) or value.dim() < 2:
        return None
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, int(value.shape[-1]))
        .contiguous()
    )


class ActivationTap:
    """One forward pass, every named module's input and output rows."""

    def __init__(self, model: torch.nn.Module, paths: Sequence[str]) -> None:
        self._model = model
        self._paths = tuple(paths)
        self.records: dict[str, dict[str, Any]] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationTap":
        for path in self._paths:
            module = self._model.get_submodule(path)
            self._handles.append(module.register_forward_hook(self._hook(path)))
        return self

    def __exit__(self, *_exception: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, path: str) -> Any:
        def hook(_module: Any, args: Any, output: Any) -> None:
            record = self.records.setdefault(
                path, {"calls": 0, "inputs": [], "outputs": [], "raw_inputs": []}
            )
            record["calls"] += 1
            # chimera's depth selectors receive the *live* ``completed_blocks``
            # list, which the trunk keeps appending to after the call returns.
            # Snapshotting the container here is the difference between reading
            # the depth sources this call actually saw and reading whatever the
            # list grew into by the end of the forward.
            record["raw_inputs"].append(
                tuple(
                    tuple(item for item in value)
                    if isinstance(value, (list, tuple))
                    else value
                    for value in args
                )
            )
            record["inputs"].append(tuple(_rows(value) for value in args))
            first = output[0] if isinstance(output, tuple) else output
            record["outputs"].append(_rows(first))

        return hook

    def input_rows(self, path: str, index: int = 0) -> torch.Tensor:
        record = self.records.get(path)
        if record is None or not record["inputs"]:
            raise ChimeraOmniFoldError(f"No captured input rows for {path!r}.")
        rows = record["inputs"][0][index]
        if rows is None:
            raise ChimeraOmniFoldError(f"Input {index} of {path!r} is not a row set.")
        return rows

    def output_rows(self, path: str) -> torch.Tensor:
        record = self.records.get(path)
        if record is None or record["outputs"][0] is None:
            raise ChimeraOmniFoldError(f"No captured output rows for {path!r}.")
        return record["outputs"][0]

    def invoked(self, path: str) -> bool:
        record = self.records.get(path)
        return bool(record and record["calls"])

    def depth_values(self, path: str) -> torch.Tensor:
        """Rebuild ``[*completed_blocks, prefix_sum]`` for a depth selector."""

        record = self.records.get(path)
        if record is None or not record["raw_inputs"]:
            raise ChimeraOmniFoldError(f"Depth selector {path!r} was not invoked.")
        prefix_sum, completed = record["raw_inputs"][0]
        stacked = torch.stack([*completed, prefix_sum], dim=2)
        return _rows(stacked)


def tap_paths(config: Any) -> tuple[str, ...]:
    paths = [
        "native_media_bridge.feature_proj",
        "native_media_bridge.position_proj",
        "native_media_bridge.norm",
        "output_attention_residual",
        "output_attention_residual.norm",
        "norm",
        "lm_head",
        "native_media_head.norm",
        "native_media_head.out",
    ]
    for index in range(int(config.num_layers)):
        block = f"blocks.{index}"
        paths.extend(
            (
                f"{block}.attention_residual",
                f"{block}.attention_residual.norm",
                f"{block}.norm1",
                f"{block}.mixer.q_proj",
                f"{block}.mixer.k_proj",
                f"{block}.mixer.v_proj",
                f"{block}.mixer.out",
                f"{block}.feed_forward_residual",
                f"{block}.feed_forward_residual.norm",
                f"{block}.norm2",
                f"{block}.ffn.up",
                f"{block}.ffn.down",
            )
        )
    for index in range(int(config.mtp_heads)):
        paths.append(f"mtp_heads.{index}")
    for name in ("flow", "grounding", "sync"):
        if name in tuple(config.aux_heads):
            paths.append(f"{name}_head")
    return tuple(paths)


def run_capture(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    paths: Sequence[str],
) -> ActivationTap:
    payload = {
        key: value for key, value in batch.items() if key not in {"native_media_targets", "native_media_loss_mask"}
    }
    tap = ActivationTap(model, paths)
    with tap:
        with torch.no_grad():
            model(**payload, return_aux=True)
    return tap


# ---------------------------------------------------------------------------
# The residual chart
# ---------------------------------------------------------------------------


def build_residual_chart(
    parent: torch.nn.Module,
    tap: ActivationTap,
    config: Any,
    *,
    target_width: int,
) -> dict[str, Any]:
    """One global orthonormal chart from the parent's own residual rows.

    Every residual-stream observation the trunk actually makes -- the depth
    selector inputs, both pre-branch norm inputs of every layer, and the final
    norm input -- is pooled, and the chart is the leading right-singular
    subspace of that pool.  ``E`` has orthonormal columns, so ``E^T`` is the
    exact left inverse on its own range and ``x -> x E`` is a projection, not a
    fit.

    This is deliberately *one* chart rather than the production
    ``ResidualBoundaryChartSet``: that artifact authenticates itself against a
    sealed ``HFCausalCircuitEvidence`` bundle, which a chimera parent cannot
    produce.  One global chart keeps every residual add inside one coordinate
    system, which is the property the per-boundary machinery has to work to
    recover, and it is stated as a bound rather than presented as equivalent.
    """

    pool: list[torch.Tensor] = []
    sources: list[str] = []
    for index in range(int(config.num_layers)):
        block = f"blocks.{index}"
        for name in (f"{block}.norm1", f"{block}.norm2"):
            pool.append(tap.input_rows(name))
            sources.append(name)
        for name in (
            f"{block}.attention_residual",
            f"{block}.feed_forward_residual",
        ):
            if tap.invoked(name):
                pool.append(tap.depth_values(name))
                sources.append(f"{name}.depth_sources")
    pool.append(tap.input_rows("norm"))
    sources.append("norm")
    rows = torch.cat(pool, dim=0)
    _u, singular_values, right = torch.linalg.svd(rows, full_matrices=False)
    encoder = right[: int(target_width)].transpose(0, 1).contiguous()
    identity_defect = float(
        (
            encoder.transpose(0, 1) @ encoder
            - torch.eye(int(target_width), dtype=torch.float64)
        )
        .abs()
        .max()
        .item()
    )
    if identity_defect > 1.0e-9:
        raise ChimeraOmniFoldError(
            f"The residual chart is not orthonormal: defect {identity_defect!r}."
        )
    energy = singular_values.square()
    retained = float(energy[: int(target_width)].sum().item())
    total = float(energy.sum().item())
    projection = (rows @ encoder) @ encoder.transpose(0, 1)
    del parent
    return {
        "encoder": encoder,
        "report": {
            "method": "single_global_orthonormal_residual_chart",
            "source_width": int(rows.shape[1]),
            "target_width": int(target_width),
            "pooled_rows": int(rows.shape[0]),
            "pooled_sources": sources,
            "encoder_sha256": tensor_sha256(encoder),
            "orthonormality_defect": identity_defect,
            "retained_energy_fraction": retained / max(total, 1.0e-300),
            "projection_relative_rmse": relative_rmse(projection, rows),
            "per_boundary_chart_set": False,
            "per_boundary_chart_set_gap": (
                "ResidualBoundaryChartSet authenticates against a sealed "
                "HFCausalCircuitEvidence bundle, which a chimera-omni parent "
                "cannot produce; the per-boundary path is a named gap."
            ),
            "parent_forward_calls_for_chart": 1,
            "training_steps": 0,
            "autograd_used": False,
        },
    }


def gap_closed(blank: float, folded: float, parent: float) -> float:
    """How much of the blank-to-parent gap the fold recovered."""

    span = float(blank) - float(parent)
    if span <= 0.0:
        return float("nan")
    return (float(blank) - float(folded)) / span


def relative_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    difference = (prediction - target).double()
    reference = target.double()
    denominator = float(reference.square().mean().sqrt().item())
    if denominator <= 0.0:
        return float("inf")
    return float(difference.square().mean().sqrt().item()) / denominator


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------


class ChildInstaller:
    """Closed-form, zero-gradient installation of one 4x narrower child."""

    def __init__(
        self,
        *,
        parent: torch.nn.Module,
        child: torch.nn.Module,
        parent_tap: ActivationTap,
        encoder: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        paths: Sequence[str],
        ridge: float,
        rms_norm_eps: float,
    ) -> None:
        self.parent = parent
        self.child = child
        self.parent_tap = parent_tap
        self.encoder = encoder
        self.batch = batch
        self.paths = tuple(paths)
        self.ridge = float(ridge)
        self.rms_norm_eps = float(rms_norm_eps)
        self.actions: list[dict[str, Any]] = []
        self.child_forward_calls = 0

    # -- primitives -----------------------------------------------------

    def _recapture(self) -> ActivationTap:
        self.child_forward_calls += 1
        return run_capture(self.child, self.batch, self.paths)

    def _target(self, path: str) -> torch.nn.Parameter:
        module_path, _, attribute = path.rpartition(".")
        module = self.child.get_submodule(module_path) if module_path else self.child
        parameter = getattr(module, attribute)
        if not isinstance(parameter, torch.nn.Parameter):
            raise ChimeraOmniFoldError(f"{path!r} is not a child parameter.")
        return parameter

    def _install(self, path: str, value: torch.Tensor, record: Mapping[str, Any]) -> None:
        parameter = self._target(path)
        if value.dim() == 2 and value.shape[0] == 1 and parameter.dim() == 1:
            # A one-row ridge solution is a vector target (the depth-selection
            # pseudo-query); the kernel returns it in matrix layout.
            value = value.reshape(-1)
        if tuple(parameter.shape) != tuple(value.shape):
            raise ChimeraOmniFoldError(
                f"Installed tensor for {path!r} has shape "
                f"{tuple(int(size) for size in value.shape)}, expected "
                f"{tuple(int(size) for size in parameter.shape)}."
            )
        with torch.no_grad():
            parameter.copy_(value.to(dtype=parameter.dtype))
        self.actions.append(
            {
                "target_parameter": path,
                "target_shape": [int(size) for size in parameter.shape],
                "installed_sha256": tensor_sha256(parameter),
                **dict(record),
            }
        )

    def _metrics(self, metrics: Any) -> dict[str, Any]:
        return {
            "rows": int(metrics.rows),
            "output_width": int(metrics.output_width),
            "relative_rmse": float(metrics.relative_rmse),
            "mean_cosine_similarity": float(metrics.mean_cosine_similarity),
        }

    def chart_transport(self, path: str, source: torch.Tensor, note: str) -> None:
        """Exact right-multiplication by the chart, for one-hot-fed tables."""

        value = source.detach().to(dtype=torch.float64) @ self.encoder
        self._install(
            path,
            value,
            {
                "method": "chart_transport_exact",
                "note": note,
                "fit_rows": 0,
            },
        )

    def ridge_linear(
        self,
        path: str,
        input_rows: torch.Tensor,
        target_rows: torch.Tensor,
        note: str,
    ) -> dict[str, Any]:
        result = fit_ridge_linear(
            input_rows,
            target_rows,
            ridge=self.ridge,
            fit_bias=False,
        )
        metrics = self._metrics(result.train_metrics)
        self._install(
            path,
            result.weight,
            {
                "method": "closed_form_ridge_linear",
                "kernel": "neural_forge.hf_closed_form_fit.fit_ridge_linear",
                "ridge": self.ridge,
                "note": note,
                "fit_rows": metrics["rows"],
                "fit": metrics,
            },
        )
        return metrics

    def diagonal_norm(
        self,
        path: str,
        input_rows: torch.Tensor,
        target_rows: torch.Tensor,
        note: str,
    ) -> dict[str, Any]:
        result = fit_diagonal_norm(
            input_rows,
            target_rows,
            kind="rms_norm",
            epsilon=self.rms_norm_eps,
            has_bias=False,
            ridge=self.ridge,
        )
        metrics = self._metrics(result.train_metrics)
        self._install(
            path,
            result.weight,
            {
                "method": "closed_form_diagonal_rms_norm",
                "kernel": "neural_forge.hf_closed_form_fit.fit_diagonal_norm",
                "ridge": self.ridge,
                "epsilon": self.rms_norm_eps,
                "note": note,
                "fit_rows": metrics["rows"],
                "fit": metrics,
            },
        )
        return metrics

    def unexercised(self, path: str, value: torch.Tensor, note: str) -> None:
        self._install(
            path,
            value,
            {
                "method": "unexercised_module_deterministic_install",
                "note": note,
                "fit_rows": 0,
            },
        )

    # -- the depth-axis selector ----------------------------------------

    def install_depth_selector(self, module_path: str, layer_label: str) -> None:
        parent_module = self.parent.get_submodule(module_path)
        if not self.parent_tap.invoked(module_path):
            # ``block_attnres`` never invokes the first block's attention
            # selector: no block has completed at that depth.  The tensors are
            # dead weight there, so they are installed deterministically and
            # said to be so rather than fitted against rows that do not exist.
            self.unexercised(
                f"{module_path}.norm.weight",
                torch.ones(
                    int(self.encoder.shape[1]), dtype=torch.float64
                ),
                f"{layer_label} depth selector is never invoked at this depth",
            )
            self.unexercised(
                f"{module_path}.query",
                self.encoder.transpose(0, 1)
                @ parent_module.query.detach().to(dtype=torch.float64),
                f"{layer_label} depth selector is never invoked at this depth",
            )
            return
        parent_values = self.parent_tap.depth_values(module_path)
        parent_keys = self.parent_tap.output_rows(f"{module_path}.norm")
        child_tap = self._recapture()
        child_values = child_tap.depth_values(module_path)
        self.diagonal_norm(
            f"{module_path}.norm.weight",
            child_values,
            parent_keys @ self.encoder,
            f"{layer_label} depth-source norm",
        )
        parent_query = parent_module.query.detach().to(dtype=torch.float64)
        parent_scores = (parent_keys @ parent_query).reshape(-1, 1)
        child_keys = normalize_rows(
            child_values, kind="rms_norm", epsilon=self.rms_norm_eps
        ) * self._target(f"{module_path}.norm.weight").detach().to(
            dtype=torch.float64
        )
        self.ridge_linear(
            f"{module_path}.query",
            child_keys,
            parent_scores,
            f"{layer_label} depth-selection pseudo-query",
        )
        del parent_values

    # -- the whole model -------------------------------------------------

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        encoder = self.encoder
        layers = len(self.child.blocks)

        # 1. Token and media-type tables: their inputs are one-hot, so the
        #    chart transport is exactly optimal and no fit is needed.
        self.chart_transport(
            "embed.weight",
            self.parent.embed.weight,
            "token embedding rows are residual vectors",
        )
        self.chart_transport(
            "native_media_bridge.type_embed.weight",
            self.parent.native_media_bridge.type_embed.weight,
            "typed-ledger media type rows are residual vectors",
        )

        # 2. The encoder-free boundary projection: raw continuous patches enter
        #    both models identically, so the input rows are shared and only the
        #    residual-side coordinates change.
        self.ridge_linear(
            "native_media_bridge.feature_proj.weight",
            self.parent_tap.input_rows("native_media_bridge.feature_proj"),
            self.parent_tap.output_rows("native_media_bridge.feature_proj") @ encoder,
            "shared native media boundary projection",
        )
        self.ridge_linear(
            "native_media_bridge.position_proj.weight",
            self.parent_tap.input_rows("native_media_bridge.position_proj"),
            self.parent_tap.output_rows("native_media_bridge.position_proj") @ encoder,
            "continuous native position projection",
        )
        child_tap = self._recapture()
        self.diagonal_norm(
            "native_media_bridge.norm.weight",
            child_tap.input_rows("native_media_bridge.norm"),
            self.parent_tap.output_rows("native_media_bridge.norm") @ encoder,
            "native media bridge norm",
        )

        # 3. The trunk, in execution order, recapturing the child between
        #    groups so every fit sees the rows the child will really produce.
        for index in range(layers):
            block = f"blocks.{index}"
            label = f"layer {index}"
            self.install_depth_selector(f"{block}.attention_residual", label)
            child_tap = self._recapture()
            self.diagonal_norm(
                f"{block}.norm1.weight",
                child_tap.input_rows(f"{block}.norm1"),
                self.parent_tap.output_rows(f"{block}.norm1") @ encoder,
                f"{label} pre-operator norm",
            )
            child_tap = self._recapture()
            normalized = child_tap.output_rows(f"{block}.norm1")
            for projection in ("q_proj", "k_proj", "v_proj"):
                self.ridge_linear(
                    f"{block}.mixer.{projection}.weight",
                    normalized,
                    self.parent_tap.output_rows(f"{block}.mixer.{projection}"),
                    f"{label} local attention {projection}",
                )
            child_tap = self._recapture()
            self.ridge_linear(
                f"{block}.mixer.out.weight",
                child_tap.input_rows(f"{block}.mixer.out"),
                self.parent_tap.output_rows(f"{block}.mixer.out") @ encoder,
                f"{label} local attention residual write",
            )
            self.install_depth_selector(f"{block}.feed_forward_residual", label)
            child_tap = self._recapture()
            self.diagonal_norm(
                f"{block}.norm2.weight",
                child_tap.input_rows(f"{block}.norm2"),
                self.parent_tap.output_rows(f"{block}.norm2") @ encoder,
                f"{label} pre-feed-forward norm",
            )
            child_tap = self._recapture()
            self.ridge_linear(
                f"{block}.ffn.up.weight",
                child_tap.output_rows(f"{block}.norm2"),
                self.parent_tap.output_rows(f"{block}.ffn.up"),
                f"{label} fused gate/up projection",
            )
            child_tap = self._recapture()
            self.ridge_linear(
                f"{block}.ffn.down.weight",
                child_tap.input_rows(f"{block}.ffn.down"),
                self.parent_tap.output_rows(f"{block}.ffn.down") @ encoder,
                f"{label} feed-forward residual write",
            )

        # 4. The trunk output selector and the final norm.
        self.install_depth_selector("output_attention_residual", "trunk output")
        child_tap = self._recapture()
        self.diagonal_norm(
            "norm.weight",
            child_tap.input_rows("norm"),
            self.parent_tap.output_rows("norm") @ encoder,
            "trunk final norm",
        )

        # 5. Every head, against the parent's own outputs.
        child_tap = self._recapture()
        hidden = child_tap.output_rows("norm")
        self.ridge_linear(
            "lm_head.weight",
            hidden,
            self.parent_tap.output_rows("lm_head"),
            "text output head",
        )
        for index in range(len(self.child.mtp_heads)):
            self.ridge_linear(
                f"mtp_heads.{index}.weight",
                hidden,
                self.parent_tap.output_rows(f"mtp_heads.{index}"),
                f"multi-token prediction head {index}",
            )
        for name in ("flow", "grounding", "sync"):
            if getattr(self.child, f"{name}_head", None) is None:
                continue
            self.ridge_linear(
                f"{name}_head.weight",
                hidden,
                self.parent_tap.output_rows(f"{name}_head"),
                f"{name} auxiliary head",
            )

        # 6. The shared native media emission head: the second real
        #    MODALITY_OUTPUT emitter in this repository.
        child_tap = self._recapture()
        self.diagonal_norm(
            "native_media_head.norm.weight",
            child_tap.input_rows("native_media_head.norm"),
            self.parent_tap.output_rows("native_media_head.norm") @ encoder,
            "native media emission norm",
        )
        child_tap = self._recapture()
        self.ridge_linear(
            "native_media_head.out.weight",
            child_tap.output_rows("native_media_head.norm"),
            self.parent_tap.output_rows("native_media_head.out"),
            "shared native media emission head",
        )
        return {
            "actions": self.actions,
            "action_count": len(self.actions),
            "child_forward_calls": self.child_forward_calls,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }


# ---------------------------------------------------------------------------
# Behavior evidence
# ---------------------------------------------------------------------------


def score_model(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    with torch.no_grad():
        output = model(**batch, return_aux=True)
    reconstruction = output["aux"]["native_media_reconstruction"].double()
    targets = batch["native_media_targets"].double()
    mask = batch["native_media_loss_mask"].double()
    error = (reconstruction - targets).square().mean(dim=-1)
    media_mse = float(
        (error * mask).sum().item() / max(1.0, float(mask.sum().item()))
    )
    logits = output["logits"].double()
    labels = batch["input_ids"][:, 1:].reshape(-1)
    flat = logits[:, :-1].reshape(-1, logits.shape[-1])
    text_ce = float(
        torch.nn.functional.cross_entropy(flat, labels).item()
    )
    return {
        "native_media_masked_mse": media_mse,
        "text_cross_entropy": text_ce,
        "text_perplexity": float(math.exp(min(text_ce, 60.0))),
        "logits": logits,
        "reconstruction": reconstruction,
        "mask": mask,
    }


def compare_to_parent(
    parent_score: Mapping[str, Any],
    child_score: Mapping[str, Any],
) -> dict[str, Any]:
    parent_top1 = parent_score["logits"].argmax(dim=-1)
    child_top1 = child_score["logits"].argmax(dim=-1)
    agreement = float((parent_top1 == child_top1).double().mean().item())
    mask = parent_score["mask"]
    selected = mask > 0.0
    parent_media = parent_score["reconstruction"][selected]
    child_media = child_score["reconstruction"][selected]
    return {
        "next_token_top1_agreement": agreement,
        "emitted_media_relative_rmse_vs_parent": relative_rmse(
            child_media, parent_media
        ),
        "emitted_media_rows_compared": int(parent_media.shape[0]),
    }


def render_emissions(
    tiny_proof: Any,
    media_render: Any,
    model: torch.nn.Module,
    config: Any,
    *,
    directory: Path,
    seq_len: int,
    max_media_tokens: int,
    emission_seed: int,
) -> dict[str, Any]:
    """Render what the model actually emits, with chimera-omni's renderers.

    ``make_sample`` draws the generation-row hint from the global RNG, so the
    seed is reset before every stimulus.  Without that the parent, the folded
    child and the blank child would each be answering a different question and
    their reconstruction errors would not be comparable.  The stimulus digest
    below is what proves all three saw the same input.
    """

    directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    paths: list[Path] = []
    stimulus: list[list[str]] = []
    plan = (
        ("image", 2),
        ("video", 4),
        ("audio", 1),
        ("music", 5),
    )
    for index, (modality, style) in enumerate(plan):
        torch.manual_seed(int(emission_seed) + index)
        sample = tiny_proof.make_sample(
            modality=modality,
            style=style,
            cfg=config,
            seq_len=int(seq_len),
            max_media_tokens=int(max_media_tokens),
            device=torch.device("cpu"),
        )
        stimulus.append(
            [
                modality,
                tensor_sha256(sample.input_ids.to(dtype=torch.float64)),
                tensor_sha256(sample.native_features),
                tensor_sha256(sample.native_targets),
            ]
        )
        with torch.no_grad():
            output = model(
                sample.input_ids.unsqueeze(0),
                native_media_features=sample.native_features.unsqueeze(0),
                native_media_type_ids=sample.type_ids.unsqueeze(0),
                native_media_positions=sample.positions.unsqueeze(0),
                native_media_indices=sample.indices.unsqueeze(0),
                return_aux=True,
            )
        count = tiny_proof.media_token_count(modality)
        emitted = (
            output["aux"]["native_media_reconstruction"][0][:count]
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        target = sample.native_targets[:count].detach().cpu().float().numpy()
        if modality == "image":
            path = directory / "emitted_image.png"
            media_render.render_image_features(emitted, path)
            paths.append(path)
            artifacts["image"] = str(path)
        elif modality == "video":
            frames = directory / "emitted_video_frames"
            manifest = media_render.render_video_features(emitted, frames, frames=4)
            paths.extend(Path(value) for value in manifest["frames"])
            paths.append(frames / "video_manifest.json")
            if manifest.get("mp4"):
                paths.append(Path(str(manifest["mp4"])))
            artifacts["video"] = manifest
        elif modality == "audio":
            path = directory / "emitted_audio.wav"
            media_render.render_wave_segments(emitted, path, seconds=1.0)
            paths.append(path)
            artifacts["audio"] = str(path)
        else:
            path = directory / "emitted_music.wav"
            media_render.render_wave_segments(emitted, path, seconds=1.5)
            paths.append(path)
            artifacts["music"] = str(path)
        artifacts[f"{modality}_target_relative_rmse"] = relative_rmse(
            torch.as_tensor(emitted, dtype=torch.float64),
            torch.as_tensor(target, dtype=torch.float64),
        )
    sizes = media_render.artifact_sizes(paths)
    artifacts["sizes"] = {
        Path(name).name: int(value) for name, value in sizes.items()
    }
    artifacts["digests"] = {
        path.name: file_sha256(path) for path in paths if path.is_file()
    }
    artifacts["empty_files"] = sorted(
        path.name for path in paths if path.is_file() and path.stat().st_size == 0
    )
    artifacts["stimulus_sha256"] = canonical_sha256(
        {"kind": "chimera_emission_stimulus", "rows": stimulus}
    )
    artifacts["emission_seed"] = int(emission_seed)
    return artifacts


# ---------------------------------------------------------------------------
# Coverage matrix
# ---------------------------------------------------------------------------


def coverage_rows(graph: Any) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for node in graph.nodes:
        entry = rows.setdefault(
            node.kind,
            {
                "circuit_kind": node.kind,
                "nodes": 0,
                "logical_parameters": 0,
                "operator_types": set(),
                "fold_kernel_families": set(),
            },
        )
        entry["nodes"] += 1
        entry["logical_parameters"] += node.logical_parameter_count
        attributes = dict(node.attributes)
        operator = attributes.get("operator_type") or attributes.get("channel_mixer")
        if isinstance(operator, str) and operator:
            entry["operator_types"].add(operator)
        kernel = attributes.get("fold_kernel_family_required")
        if isinstance(kernel, str) and kernel:
            entry["fold_kernel_families"].add(kernel)
    result = []
    for kind in sorted(rows):
        entry = rows[kind]
        result.append(
            {
                "circuit_kind": entry["circuit_kind"],
                "nodes": entry["nodes"],
                "logical_parameters": entry["logical_parameters"],
                "operator_types": sorted(entry["operator_types"]),
                "fold_kernel_families": sorted(entry["fold_kernel_families"]),
            }
        )
    return result


def parameter_map(parent: torch.nn.Module, child: torch.nn.Module) -> list[dict[str, Any]]:
    parent_named = dict(parent.named_parameters(recurse=True, remove_duplicate=False))
    child_named = dict(child.named_parameters(recurse=True, remove_duplicate=False))
    if set(parent_named) != set(child_named):
        raise ChimeraOmniFoldError(
            "Parent and child do not expose the same parameter names; the "
            "compaction is not a pure residual-width contraction."
        )
    rows = []
    for name in sorted(parent_named):
        source = parent_named[name]
        target = child_named[name]
        rows.append(
            {
                "parameter": name,
                "parent_shape": [int(size) for size in source.shape],
                "child_shape": [int(size) for size in target.shape],
                "parent_elements": int(source.numel()),
                "child_elements": int(target.numel()),
                "ratio_numerator": int(source.numel()),
                "ratio_denominator": int(target.numel()),
                "exactly_four_times": int(source.numel()) == 4 * int(target.numel()),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fold a real tiny chimera-omni model at exact 4x compaction."
    )
    parser.add_argument("--chimera-omni-repo", default=None)
    parser.add_argument(
        "--out", default="reports/chimera-omni-fold-readiness-20260804"
    )
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--calibration-batch-size", type=int, default=24)
    parser.add_argument("--holdout-batch-size", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-media-tokens", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--parent-seed", type=int, default=7)
    parser.add_argument("--child-seed", type=int, default=11)
    parser.add_argument("--calibration-seed", type=int, default=101)
    parser.add_argument("--holdout-seed", type=int, default=202)
    parser.add_argument("--emission-seed", type=int, default=303)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.out)
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    repo = resolve_chimera_omni_repo(args.chimera_omni_repo)
    modules = import_chimera_omni(repo)
    chimera_config = modules["config"]
    chimera_modeling = modules["modeling"]
    tiny_proof = modules["tiny_proof"]
    media_render = modules["media_render"]

    parent_config = coverable_config(chimera_config, hidden_size=PARENT_HIDDEN_SIZE)
    child_config = coverable_config(chimera_config, hidden_size=CHILD_HIDDEN_SIZE)

    # -- configuration --------------------------------------------------
    configuration_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "configuration",
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "torch_version": str(torch.__version__),
        "python_version": sys.version.split()[0],
        "torch_threads": int(torch.get_num_threads()),
        "chimera_omni": chimera_source_identity(repo),
        "compaction_factor": COMPACTION_FACTOR,
        "compaction_axis": "residual_hidden_width_only",
        "parent_config": parent_config.to_manifest(),
        "child_config": child_config.to_manifest(),
        "avoided_block_families": list(AVOIDED_BLOCK_FAMILIES),
        "avoided_channel_mixers": list(AVOIDED_CHANNEL_MIXERS),
        "avoided_reason": (
            "no fold kernel exists for these families; the inspector compiles "
            "them, the planner refuses them, and this run does not attempt them"
        ),
        "sequence_length": int(args.sequence_length),
        "max_media_tokens": int(args.max_media_tokens),
        "ridge": float(args.ridge),
        "seeds": {
            "parent": int(args.parent_seed),
            "child": int(args.child_seed),
            "calibration": int(args.calibration_seed),
            "holdout": int(args.holdout_seed),
            "emission": int(args.emission_seed),
        },
    }
    configuration_body["input_sha256"] = canonical_sha256(configuration_body)
    configuration = write_receipt(root / "configuration.json", configuration_body)

    # -- parent ----------------------------------------------------------
    torch.manual_seed(int(args.parent_seed))
    parent = chimera_modeling.ChimeraOmniModel(parent_config)
    blank_parent_fingerprint = fingerprint_model_state(parent)
    training = train_parent(
        tiny_proof,
        parent,
        parent_config,
        steps=int(args.train_steps),
        batch_size=int(args.train_batch_size),
        seq_len=int(args.sequence_length),
        max_media_tokens=int(args.max_media_tokens),
        learning_rate=float(args.train_learning_rate),
        seed=int(args.parent_seed),
    )
    parent.eval()
    torch.manual_seed(int(args.child_seed))
    child = chimera_modeling.ChimeraOmniModel(child_config)
    child.eval()
    blank_child = chimera_modeling.ChimeraOmniModel(child_config)
    blank_child.load_state_dict(child.state_dict())
    blank_child.eval()
    blank_child_fingerprint = fingerprint_model_state(child)
    parent_fingerprint = fingerprint_model_state(parent)
    if parent_fingerprint == blank_parent_fingerprint:
        raise ChimeraOmniFoldError("Training did not change the parent weights.")
    parent_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "parent",
        "input_sha256": configuration["input_sha256"],
        "parent_unique_parameters": unique_parameter_count(parent),
        "child_unique_parameters": unique_parameter_count(child),
        "blank_parent_state_sha256": blank_parent_fingerprint,
        "trained_parent_state_sha256": parent_fingerprint,
        "blank_child_state_sha256": blank_child_fingerprint,
        "training": training,
    }
    write_receipt(root / "parent.json", parent_body)

    # -- inspection ------------------------------------------------------
    inspector = ChimeraOmniCircuitInspector()
    parent_graph = inspect_model_circuits(parent, model_name="chimera-omni-parent")
    child_graph = inspect_model_circuits(child, model_name="chimera-omni-child")
    for graph in (parent_graph, child_graph):
        if dict(graph.metadata)["inspector"] != inspector.name:
            raise ChimeraOmniFoldError(
                "The chimera graphs were not compiled by the chimera inspector."
            )
    parent_paths = set(parent_graph.parameter_paths)
    live_paths = {
        name
        for name, _value in parent.named_parameters(recurse=True, remove_duplicate=False)
    }
    if parent_paths != live_paths:
        raise ChimeraOmniFoldError(
            "The inspected graph does not own exactly the model's tensors."
        )
    emitted = emitted_media_interfaces(parent_graph)
    inspection_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "inspection",
        "input_sha256": configuration["input_sha256"],
        "inspector": inspector.name,
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "child_graph_fingerprint": child_graph.fingerprint,
        "parent_structural_topology_fingerprint": (
            parent_graph.structural_topology_fingerprint
        ),
        "child_structural_topology_fingerprint": (
            child_graph.structural_topology_fingerprint
        ),
        "parent_metadata": dict(parent_graph.metadata),
        "child_metadata": dict(child_graph.metadata),
        "parent_node_count": len(parent_graph.nodes),
        "parent_edge_count": len(parent_graph.edges),
        "parent_interface_count": len(parent_graph.interfaces),
        "complete_parameter_ownership": True,
        "coverage_matrix": coverage_rows(parent_graph),
        "input_interfaces": [
            value.as_dict()
            for value in parent_graph.interfaces
            if value.direction == "input"
        ],
        "output_interfaces": [
            value.as_dict()
            for value in parent_graph.interfaces
            if value.direction == "output"
        ],
        "emitted_media_interfaces": [value.as_dict() for value in emitted],
        "emitted_media_interface_count": len(emitted),
        "modality_output_nodes": [
            node.node_id
            for node in parent_graph.nodes_of_kind(CircuitKind.MODALITY_OUTPUT)
        ],
        "modality_input_nodes": [
            node.node_id
            for node in parent_graph.nodes_of_kind(CircuitKind.MODALITY_INPUT)
        ],
        "depth_residual_nodes": [
            node.node_id
            for node in parent_graph.nodes_of_kind(CircuitKind.RESIDUAL_ADD)
            if dict(node.attributes).get("residual_mode") == "block_attnres"
        ],
    }
    inspection = write_receipt(root / "inspection.json", inspection_body)

    # -- budget ----------------------------------------------------------
    parent_unique = parent_graph.unique_parameter_count
    child_unique = child_graph.unique_parameter_count
    exact = parent_unique == COMPACTION_FACTOR * child_unique
    rows = parameter_map(parent, child)
    budget_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "budget",
        "input_sha256": configuration["input_sha256"],
        "parent_unique_parameters": parent_unique,
        "child_unique_parameters": child_unique,
        "parent_logical_parameters": parent_graph.logical_parameter_count,
        "child_logical_parameters": child_graph.logical_parameter_count,
        "requested_factor": COMPACTION_FACTOR,
        "observed_factor_numerator": parent_unique,
        "observed_factor_denominator": child_unique,
        "exact_four_times": bool(exact),
        "residual_from_exact": parent_unique - COMPACTION_FACTOR * child_unique,
        "quantization": False,
        "sparsity": False,
        "storage_sharing": False,
        "parent_sidecar": False,
        "tensor_count": len(rows),
        "tensors_exactly_four_times": sum(
            1 for row in rows if row["exactly_four_times"]
        ),
        "parameter_map": rows,
    }
    if not exact:
        write_receipt(root / "budget.json", budget_body)
        raise ChimeraOmniFoldError(
            f"The child is {parent_unique}/{child_unique} of the parent, not "
            f"exactly {COMPACTION_FACTOR}x."
        )
    write_receipt(root / "budget.json", budget_body)

    # -- plan ------------------------------------------------------------
    plan_record: dict[str, Any]
    try:
        plan = DynamicFoldPlanner(
            enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS
        ).plan(
            (parent_graph,),
            child_graph,
            child_state_fingerprint=blank_child_fingerprint,
            evidence_fingerprint=canonical_sha256(
                {"kind": "chimera_toy_fold_calibration", "seed": args.calibration_seed}
            ),
            parent_state_fingerprints={"chimera-omni-parent": parent_fingerprint},
        )
        plan_record = {
            "planner_ready": bool(plan.ready),
            "action_count": len(plan.actions),
            "action_methods": sorted({value.method_id for value in plan.actions}),
            "gap_count": len(plan.gaps),
            "gaps": sorted({value.required_capability for value in plan.gaps}),
        }
    except DynamicFoldPlanningError as exc:
        plan_record = {
            "planner_ready": False,
            "planning_error": str(exc),
            "action_count": 0,
            "action_methods": [],
            "gap_count": None,
            "gaps": [],
        }
    plan_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "plan",
        "input_sha256": configuration["input_sha256"],
        "enabled_experimental_method_ids": list(CAUSAL_LAYER_BRANCH_METHODS),
        **plan_record,
        "interpretation": (
            "The promoted causal branch kernels require both graphs to declare "
            "inspector='hf-causal-decoder' and position_policy='rope'. A "
            "chimera-omni graph declares neither, so the planner refuses this "
            "crossing rather than emitting a transaction no registered operator "
            "could execute. That refusal is the correct behavior and it is why "
            "the numeric fold below is a driver-local closed-form transaction "
            "built from hf_closed_form_fit rather than a planned action."
        ),
    }
    write_receipt(root / "plan.json", plan_body)

    # -- fold -------------------------------------------------------------
    calibration, calibration_specs = build_batch(
        tiny_proof,
        parent_config,
        batch_size=int(args.calibration_batch_size),
        seq_len=int(args.sequence_length),
        max_media_tokens=int(args.max_media_tokens),
        seed=int(args.calibration_seed),
    )
    holdout, holdout_specs = build_batch(
        tiny_proof,
        parent_config,
        batch_size=int(args.holdout_batch_size),
        seq_len=int(args.sequence_length),
        max_media_tokens=int(args.max_media_tokens),
        seed=int(args.holdout_seed),
    )
    paths = tap_paths(parent_config)
    parent_tap = run_capture(parent, calibration, paths)
    chart = build_residual_chart(
        parent, parent_tap, parent_config, target_width=CHILD_HIDDEN_SIZE
    )
    installer = ChildInstaller(
        parent=parent,
        child=child,
        parent_tap=parent_tap,
        encoder=chart["encoder"],
        batch=calibration,
        paths=paths,
        ridge=float(args.ridge),
        rms_norm_eps=float(parent_config.rms_norm_eps),
    )
    try:
        install_report, guard = run_fold_only_callback(
            installer.run,
            phase="chimera-omni-toy-fold",
            protected_models=(parent,),
        )
    except DynamicFoldExecutionError as exc:
        raise ChimeraOmniFoldError(f"The fold-only guard refused the run: {exc}") from exc
    if not guard.clean:
        raise ChimeraOmniFoldError(
            f"The fold performed training operations: {guard.as_dict()!r}"
        )
    folded_fingerprint = fingerprint_model_state(child)
    if folded_fingerprint == blank_child_fingerprint:
        raise ChimeraOmniFoldError("The fold did not change the child weights.")
    if fingerprint_model_state(parent) != parent_fingerprint:
        raise ChimeraOmniFoldError("The fold mutated the parent.")
    installed = {row["target_parameter"] for row in install_report["actions"]}
    expected = {
        name
        for name, _value in child.named_parameters(recurse=True, remove_duplicate=False)
    }
    if installed != expected:
        raise ChimeraOmniFoldError(
            "The fold did not install every child tensor exactly once; "
            f"missing={sorted(expected - installed)!r}, "
            f"extra={sorted(installed - expected)!r}."
        )
    fold_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "fold",
        "input_sha256": configuration["input_sha256"],
        "parent_graph_fingerprint": inspection["parent_graph_fingerprint"],
        "calibration_batch_sha256": batch_digest(calibration),
        "calibration_modalities": sorted({name for name, _style in calibration_specs}),
        "holdout_batch_sha256": batch_digest(holdout),
        "holdout_modalities": sorted({name for name, _style in holdout_specs}),
        "chart": chart["report"],
        "parent_forward_calls": 1,
        "child_recapture_forward_calls": install_report["child_forward_calls"],
        "fold_only_guard": guard.as_dict(),
        "blank_child_state_sha256": blank_child_fingerprint,
        "folded_child_state_sha256": folded_fingerprint,
        "parent_state_unchanged": True,
        "installed_tensor_count": len(installed),
        "every_child_tensor_installed_once": True,
        "install_methods": sorted(
            {row["method"] for row in install_report["actions"]}
        ),
        "elapsed_sec": install_report["elapsed_sec"],
        "actions": install_report["actions"],
    }
    write_receipt(root / "fold.json", fold_body)

    # -- behavior ---------------------------------------------------------
    parent_score = score_model(parent, holdout)
    folded_score = score_model(child, holdout)
    blank_score = score_model(blank_child, holdout)
    folded_versus_parent = compare_to_parent(parent_score, folded_score)
    blank_versus_parent = compare_to_parent(parent_score, blank_score)
    emissions = {
        "parent": render_emissions(
            tiny_proof,
            media_render,
            parent,
            parent_config,
            directory=root / "emissions" / "parent",
            seq_len=int(args.sequence_length),
            max_media_tokens=int(args.max_media_tokens),
            emission_seed=int(args.emission_seed),
        ),
        "folded_child": render_emissions(
            tiny_proof,
            media_render,
            child,
            child_config,
            directory=root / "emissions" / "folded-child",
            seq_len=int(args.sequence_length),
            max_media_tokens=int(args.max_media_tokens),
            emission_seed=int(args.emission_seed),
        ),
        "blank_child": render_emissions(
            tiny_proof,
            media_render,
            blank_child,
            child_config,
            directory=root / "emissions" / "blank-child",
            seq_len=int(args.sequence_length),
            max_media_tokens=int(args.max_media_tokens),
            emission_seed=int(args.emission_seed),
        ),
    }
    stimuli = {value["stimulus_sha256"] for value in emissions.values()}
    if len(stimuli) != 1:
        raise ChimeraOmniFoldError(
            "The three models were rendered against different stimuli; their "
            "reconstruction errors are not comparable."
        )
    beats_blank_media = (
        folded_score["native_media_masked_mse"] < blank_score["native_media_masked_mse"]
    )
    beats_blank_text = (
        folded_score["text_cross_entropy"] < blank_score["text_cross_entropy"]
    )
    behavior_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "behavior",
        "input_sha256": configuration["input_sha256"],
        "holdout_batch_sha256": batch_digest(holdout),
        "scores": {
            "trained_parent": {
                "native_media_masked_mse": parent_score["native_media_masked_mse"],
                "text_cross_entropy": parent_score["text_cross_entropy"],
                "text_perplexity": parent_score["text_perplexity"],
            },
            "folded_child": {
                "native_media_masked_mse": folded_score["native_media_masked_mse"],
                "text_cross_entropy": folded_score["text_cross_entropy"],
                "text_perplexity": folded_score["text_perplexity"],
            },
            "blank_child": {
                "native_media_masked_mse": blank_score["native_media_masked_mse"],
                "text_cross_entropy": blank_score["text_cross_entropy"],
                "text_perplexity": blank_score["text_perplexity"],
            },
        },
        "folded_child_versus_parent": folded_versus_parent,
        "blank_child_versus_parent": blank_versus_parent,
        "folded_beats_blank_on_native_media": bool(beats_blank_media),
        "folded_beats_blank_on_text": bool(beats_blank_text),
        "native_media_gap_closed_fraction": gap_closed(
            blank_score["native_media_masked_mse"],
            folded_score["native_media_masked_mse"],
            parent_score["native_media_masked_mse"],
        ),
        "native_media_gap_closed_definition": (
            "(blank_mse - folded_mse) / (blank_mse - parent_mse); 1.0 is the "
            "trained parent, 0.0 is the blank child of the same size"
        ),
        "emitted_output_evidence": emissions,
        "emitted_output_nonempty": not (
            emissions["folded_child"]["empty_files"]
            or emissions["parent"]["empty_files"]
        ),
        "honest_limit": (
            "These are parent-relative and blank-relative numbers on one tiny "
            "synthetic objective. They show the fold moved real information "
            "into a quarter-size child without a gradient step. They do not "
            "show retained intelligence and no claim of that kind is made."
        ),
    }
    write_receipt(root / "behavior.json", behavior_body)

    # -- summary ----------------------------------------------------------
    complete_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "complete",
        "stages": list(STAGES),
        "input_sha256": configuration["input_sha256"],
        "chimera_omni_repository": str(repo),
        "parent_unique_parameters": parent_unique,
        "child_unique_parameters": child_unique,
        "exact_four_times": True,
        "inspector": inspector.name,
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "child_graph_fingerprint": child_graph.fingerprint,
        "emitted_media_interface_count": len(emitted),
        "chimera_tiny_proof_status": training["chimera_tiny_proof_status"],
        "planner_ready": plan_record["planner_ready"],
        "planner_gaps": plan_record["gaps"],
        "fold_only_guard_clean": bool(guard.clean),
        "folded_beats_blank_on_native_media": bool(beats_blank_media),
        "folded_beats_blank_on_text": bool(beats_blank_text),
        "next_token_top1_agreement": folded_versus_parent[
            "next_token_top1_agreement"
        ],
        "native_media_gap_closed_fraction": behavior_body[
            "native_media_gap_closed_fraction"
        ],
        "emitted_output_nonempty": behavior_body["emitted_output_nonempty"],
        "avoided_block_families": list(AVOIDED_BLOCK_FAMILIES),
        "avoided_channel_mixers": list(AVOIDED_CHANNEL_MIXERS),
    }
    write_receipt(root / "complete.json", complete_body)
    write_receipt(
        root / "manifest.json",
        {
            "schema": RUNNER_SCHEMA,
            "stage": "manifest",
            "input_sha256": configuration["input_sha256"],
            "skipped": ["manifest.json", "README.md"],
            "skipped_reason": (
                "manifest.json cannot digest itself, and README.md is written "
                "by hand after the run"
            ),
            "files": tree_manifest(root, skip=("manifest.json", "README.md")),
        },
    )
    print(json.dumps(complete_body, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
