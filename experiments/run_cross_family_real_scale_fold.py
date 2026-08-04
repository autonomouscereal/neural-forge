#!/usr/bin/env python3
"""Fold a real-scale llama parent into a blank qwen3 child, end to end.

``tests/test_cross_family_causal_fold_execution.py`` already proves the
cross-family execution path -- a llama parent, a blank qwen3 child with per-head
Q/K RMS normalization, an unrelated ByteLevel-BPE vocabulary, three promoted
operators, one :class:`DynamicFoldExecutor` run, zero parent forwards.  It
proves it at H=32 with two layers and a three-hundred-row vocabulary, which is
small enough that nothing it reports can be read as behavior.
``reports/kernel-stage2-20260801`` proves the cross-*width* boundary charts on a
real SmolLM2-135M parent, but only at two adjacent boundaries and only inside
one family.

This runner is the union of those two proofs at a size where the child can be
asked to write text: one parent, one blank qwen3-shaped child at a narrower
residual width, every branch of every layer installed through
``DynamicFoldExecutor``, and -- the part neither predecessor could give --
greedy continuations decoded from the folded child on held-out prompts and
saved verbatim for manual reading.

Three things about the crossing are worth stating before the numbers:

* The residual width contracts, so every branch runs through
  ``CrossWidthCausalBranchOperator`` against a sealed per-boundary chart set and
  a live sequential recapture of the child.
* The families differ, so the child's per-head Q/K RMS gains are fitted by the
  coupled GQA kernel rather than copied.
* The vocabularies differ, so the token interface is a sealed *pair* bridge.
  The planner emits **one** action for the interface under a simultaneous
  family and width crossing, and that action owns the embedding, the head and
  the final norm.  ``src`` has an operator for each half of that action and
  none for the whole of it, so this file composes one locally; see
  ``NOTES-W7A.md``.  Nothing under ``src`` is modified.

Evidence discipline is unchanged: one parent forward for the whole run,
captured before execution, and zero parent forwards during it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_forge.dynamic_causal_operator_promotion import (  # noqa: E402
    CausalGQAAttentionBranchOperator,
    CausalGatedMLPBranchOperator,
    CrossWidthCausalBranchOperator,
    DynamicCausalOperatorPromotionError,
    TokenInterfaceBridgeTransaction,
)
from neural_forge.dynamic_circuit_ir import CircuitKind, canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    DynamicFoldExecutionError,
    DynamicFoldExecutor,
    ExactCopyOperator,
    FoldDecision,
    FoldExecutionContext,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import (  # noqa: E402
    CAUSAL_ATTENTION_BRANCH_METHOD,
    CAUSAL_GATED_MLP_BRANCH_METHOD,
    CAUSAL_LAYER_BRANCH_METHODS,
    DynamicFoldPlanner,
)
from neural_forge.dynamic_fold_snapshot import (  # noqa: E402
    DynamicFoldSnapshotConfig,
    capture_model_state,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge.hf_arch_layout import build_decoder_layout  # noqa: E402
from neural_forge.hf_causal_circuit_evidence import (  # noqa: E402
    capture_hf_causal_circuit_evidence,
)
from neural_forge.hf_causal_cross_width_charts import (  # noqa: E402
    ATTENTION_INPUT_BOUNDARY,
    CROSS_WIDTH_BRANCH_FOLD_SCHEMA,
    TERMINAL_BOUNDARY_ID,
    CrossWidthResidualChartError,
    LiveChildBoundaryRecapture,
    ResidualBoundaryChartSet,
    child_residual_topology_sha256,
    compile_residual_boundary_chart_set,
    residual_boundary_id,
)
from neural_forge.hf_closed_form_causal_circuit_fold import (  # noqa: E402
    fit_rmsnorm_diagonal,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
)
from neural_forge.hf_tokenizer_pair_bridge import (  # noqa: E402
    TokenizerPairBridgeArtifact,
    compile_hf_tokenizer_pair_bridge,
    compile_pair_bridge_anchor_transport,
    compile_pair_bridge_length_normalized_transport,
)
from neural_forge.hf_tokenizer_prune import prune_hf_tokenizer  # noqa: E402


RUNNER_SCHEMA = "neural-forge-cross-family-real-scale-fold-runner-v1"
INTERFACE_SCHEMA = "neural-forge-chart-projected-pair-code-interface-v1"
TOKEN_INTERFACE_BRIDGE_METHOD = "token_interface_bridge"
MASKED_ROW_POLICY = "leave-uncovered-target-rows-at-incumbent-blank"
UNCOVERED_ROW_INDEX_LIMIT = 64

#: Stage names, in execution order.  Each one seals a receipt whose
#: ``input_sha256`` is the digest of everything upstream of it, so a resumed run
#: reuses a stage only when nothing it depends on has changed.
STAGES = (
    "configuration",
    "tokenizers",
    "evidence",
    "charts",
    "fold",
    "behavior",
)

#: The geometry stage 2 measured on the real llama-family parent.  Naming it in
#: the seal makes it impossible to read a synthesized-parent run as a
#: pretrained one.  ``rope_theta_kilo`` is the rotary base in thousands, which
#: keeps every geometry value an exact integer inside the configuration seal.
GEOMETRIES: Mapping[str, Mapping[str, int]] = {
    "smollm2-135m": {
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "head_dim": 64,
        "max_position_embeddings": 2048,
        "rms_norm_eps_micro": 10,
        "rope_theta_kilo": 100,
    },
    "tiny": {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 512,
        "rms_norm_eps_micro": 10,
        "rope_theta_kilo": 10,
    },
}


class RealScaleFoldError(RuntimeError):
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

    content = {
        key: value for key, value in document.items() if key != "receipt_sha256"
    }
    return document.get("receipt_sha256") == canonical_sha256(content)


def _write_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    sealed = seal_receipt(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(sealed))
    return sealed


def _read_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RealScaleFoldError(f"Receipt is not readable JSON: {path}.") from exc
    if not isinstance(document, dict):
        raise RealScaleFoldError(f"Receipt is not a JSON object: {path}.")
    if not verify_receipt(document):
        raise RealScaleFoldError(f"Receipt seal is broken: {path}.")
    return document


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path, skip: Sequence[str] = ()) -> list[dict[str, Any]]:
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
                "sha256": _file_sha256(path),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# The one operator ``src`` does not have
# ---------------------------------------------------------------------------


def _masked_rows(values: Any, mask: Any) -> Any:
    trailing = tuple(int(size) for size in values.shape[2:])
    return values.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)].contiguous()


def _single_path(node: Any, role: str, name: str) -> str:
    paths = tuple(node.parameter_paths)
    if len(paths) != 1:
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires exactly one learned {name} tensor."
        )
    return paths[0]


def _one_node(nodes: Sequence[Any], kind: CircuitKind, role: str) -> Any:
    matched = tuple(node for node in nodes if node.kind == kind.value)
    if len(matched) != 1:
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires exactly one {kind.value} node in the action."
        )
    return matched[0]


def _uncovered_record(name: str, mask: Any) -> dict[str, Any]:
    rows = mask.reshape(-1).to(torch.bool)
    uncovered = torch.nonzero(~rows, as_tuple=False).reshape(-1).tolist()
    listed = [int(value) for value in uncovered[:UNCOVERED_ROW_INDEX_LIMIT]]
    covered = [
        int(value)
        for value in torch.nonzero(rows, as_tuple=False).reshape(-1).tolist()
    ]
    return {
        f"{name}_uncovered_target_rows": len(uncovered),
        f"{name}_uncovered_target_row_indices": listed,
        f"{name}_uncovered_target_row_indices_truncated": len(uncovered) > len(listed),
        f"{name}_evidence_mask_sha256": canonical_sha256(
            {
                "schema": INTERFACE_SCHEMA,
                "kind": "pair_bridge_evidence_mask",
                "rows": int(rows.numel()),
                "covered": covered,
            }
        ),
    }


def _blend_covered_rows(incumbent: Any, compiled: Any, mask: Any, role: str) -> Any:
    """Write compiled rows only where the pair bridge actually has evidence."""

    live = incumbent.detach().to(dtype=torch.float64)
    if tuple(live.shape) != tuple(compiled.shape):
        raise DynamicCausalOperatorPromotionError(
            f"{role} compiled rows do not match the target matrix shape."
        )
    rows = mask.reshape(-1).to(torch.bool)
    if int(rows.numel()) != int(live.shape[0]):
        raise DynamicCausalOperatorPromotionError(
            f"{role} evidence mask does not cover every target row."
        )
    if not bool(rows.any().item()):
        raise DynamicCausalOperatorPromotionError(
            f"{role} pair bridge covered no target row at all."
        )
    return torch.where(rows.unsqueeze(1), compiled, live).contiguous()


class ChartProjectedPairCodeTokenHeadInterfaceOperator:
    """Install one interface action that crosses both the family and the width.

    ``PairCodeTokenInterfaceBridgeOperator`` transports an independent
    vocabulary but owns only the embedding and the head.
    ``CrossWidthTokenHeadInterfaceOperator`` owns the embedding, the head and
    the final norm that shares their chart pair, but refuses anything except one
    shared tokenizer identity.  When the tokenizers *and* the residual width
    both change, the planner emits a single ``token_interface_bridge`` action
    carrying all three tensors, and neither operator can execute it: one fails
    the installed-path contract, the other fails the tokenizer-identity clause.

    This composition changes no math.  Rows come from the sealed pair bridge
    under the same anchor-preferred blend and the same masked-row policy the
    promoted pair operator uses; coordinates come from the two interface-locked
    charts, which ``ResidualBoundaryChart`` already declares usable as dense
    vocabulary projectors; the final-norm gain is the same
    ``fit_rmsnorm_diagonal`` solve on chart-encoded parent states.  It lives
    here rather than in ``src`` because this task owns new files only.
    """

    method_id = TOKEN_INTERFACE_BRIDGE_METHOD

    def __init__(
        self,
        *,
        artifact: TokenizerPairBridgeArtifact,
        chart_set: ResidualBoundaryChartSet,
        ridge: float = 1e-6,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        if not isinstance(artifact, TokenizerPairBridgeArtifact):
            raise DynamicCausalOperatorPromotionError(
                "artifact must be a TokenizerPairBridgeArtifact."
            )
        if not isinstance(chart_set, ResidualBoundaryChartSet):
            raise DynamicCausalOperatorPromotionError(
                "chart_set must be a sealed ResidualBoundaryChartSet."
            )
        artifact.verify_files()
        self.artifact = artifact
        self.chart_set = chart_set.verify()
        self.ridge = float(ridge)
        self.snapshot_config = snapshot_config or DynamicFoldSnapshotConfig.memory()

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": INTERFACE_SCHEMA,
                "operator": type(self).__qualname__,
                "method_id": self.method_id,
                "bridge_artifact_sha256": self.artifact.artifact_sha256,
                "pair_profile_fingerprint": self.artifact.profile_fingerprint,
                "chart_set_sha256": self.chart_set.set_sha256,
                "masked_row_policy": MASKED_ROW_POLICY,
                "ridge": self.ridge,
                "snapshot_backend": self.snapshot_config.backend,
            }
        )

    def _pair_evidence(self, source_weight: Any) -> tuple[Any, Any, list[str]]:
        """Anchor-preferred blend: exact anchor rows, averaged code elsewhere."""

        anchor = compile_pair_bridge_anchor_transport(source_weight, self.artifact)
        averaged = compile_pair_bridge_length_normalized_transport(
            source_weight, self.artifact
        )
        anchored = anchor.evidence_mask.to(torch.bool)
        covered = averaged.evidence_mask.to(torch.bool)
        if bool((anchored & ~covered).any().item()):
            raise DynamicCausalOperatorPromotionError(
                "An anchor-exact pair transport reached a row the "
                "length-normalized transport did not; the sealed codes are "
                "inconsistent."
            )
        evidence = torch.where(
            anchored.unsqueeze(1), anchor.evidence, averaged.evidence
        )
        return (
            evidence,
            covered,
            [anchor.report["report_sha256"], averaged.report["report_sha256"]],
        )

    def prepare(self, action: Any, context: Any, evidence: Any) -> Any:
        role = "Chart-projected pair-code token/head interface operator"
        if action.method_id != self.method_id:
            raise DynamicCausalOperatorPromotionError(
                f"{role} received method {action.method_id!r}."
            )
        if dict(action.configuration).get("boundary_chart_required") is not True:
            raise DynamicCausalOperatorPromotionError(
                f"{role} only executes an action the planner marked "
                "boundary_chart_required."
            )
        if len(action.sources) != 1:
            raise DynamicCausalOperatorPromotionError(
                f"{role} accepts exactly one parent source."
            )
        parent_name = action.sources[0].parent_name
        parent_graph = context.parent_graphs[parent_name]
        child_graph = context.child_graph
        artifact = self.artifact
        artifact.verify_files()
        if artifact.parent_tokenizer_semantics_sha256 != (
            parent_graph.tokenizer_fingerprint
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} sealed source tokenizer differs from the planned "
                "parent tokenizer identity."
            )
        if artifact.child_tokenizer_semantics_sha256 != (
            child_graph.tokenizer_fingerprint
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} sealed target tokenizer differs from the planned "
                "child tokenizer identity."
            )
        sealed = evidence.payload
        if sealed is None or sealed.parent_graph_fingerprint != (
            parent_graph.fingerprint
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} evidence is bound to a different parent graph."
            )
        sealed.verify()
        layout = build_decoder_layout(context.child_model)
        try:
            chart_set = self.chart_set.authenticate(
                parent_evidence_sha256=sealed.cache_key,
                child_topology_sha256=child_residual_topology_sha256(layout),
                role=role,
            )
            embedding_chart = chart_set.chart(
                residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY)
            )
            terminal_chart = chart_set.chart(TERMINAL_BOUNDARY_ID)
        except CrossWidthResidualChartError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} chart set failed authentication: {exc}"
            ) from exc
        if not embedding_chart.interface_locked or not terminal_chart.interface_locked:
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires both interface boundaries to be chart locked."
            )
        final_norm = layout.final_norm
        if (
            final_norm.kind != "rms_norm"
            or final_norm.weight_kind != "direct_scale"
            or final_norm.has_bias
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires a bias-free direct-scale RMSNorm final norm."
            )
        source_nodes = tuple(
            parent_graph.node(node_id) for node_id in action.sources[0].node_ids
        )
        target_nodes = tuple(
            child_graph.node(node_id) for node_id in action.target_node_ids
        )
        source_embedding_node = _one_node(
            source_nodes, CircuitKind.TOKEN_EMBEDDING, role
        )
        source_head_node = _one_node(source_nodes, CircuitKind.OUTPUT_HEAD, role)
        target_embedding_node = _one_node(
            target_nodes, CircuitKind.TOKEN_EMBEDDING, role
        )
        target_head_node = _one_node(target_nodes, CircuitKind.OUTPUT_HEAD, role)
        target_embedding_path = _single_path(
            target_embedding_node, role, "child embedding"
        )
        target_head_path = _single_path(target_head_node, role, "child output head")
        final_norm_path = final_norm.path + ".weight"
        tied = bool(
            {value.storage_group for value in target_embedding_node.parameters}
            & {value.storage_group for value in target_head_node.parameters}
        )
        if tied:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot install one tied tensor from two different "
                "interface charts."
            )
        paths = tuple(
            sorted((target_embedding_path, target_head_path, final_norm_path))
        )
        if paths != tuple(sorted(action.target_parameter_paths)):
            raise DynamicCausalOperatorPromotionError(
                f"{role} layout paths differ from the planned target parameters: "
                f"layout={list(paths)}, "
                f"planned={list(action.target_parameter_paths)}."
            )
        parent_model = context.parent_models[parent_name]
        source_embedding = parent_model.get_parameter(
            _single_path(source_embedding_node, role, "parent embedding")
        )
        source_head = parent_model.get_parameter(
            _single_path(source_head_node, role, "parent output head")
        )
        target_embedding = context.child_model.get_parameter(target_embedding_path)
        target_head = context.child_model.get_parameter(target_head_path)
        profile = artifact.profile
        if (
            int(source_embedding.shape[0]) != profile.source_row_capacity
            or int(source_head.shape[0]) != profile.source_row_capacity
            or int(target_embedding.shape[0]) != profile.target_row_capacity
            or int(target_head.shape[0]) != profile.target_row_capacity
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} model rows differ from the sealed pair row capacities."
            )
        embedding_evidence, embedding_mask, embedding_reports = self._pair_evidence(
            source_embedding
        )
        head_evidence, head_mask, head_reports = self._pair_evidence(source_head)
        embedding_rows = embedding_chart.fold_consumer_weight(embedding_evidence)
        head_rows = terminal_chart.fold_consumer_weight(head_evidence)
        fit = sealed.fit_token_mask
        norm_fold = fit_rmsnorm_diagonal(
            _masked_rows(terminal_chart.encode(sealed.layers[-1].block_output), fit),
            _masked_rows(terminal_chart.encode(sealed.final_norm_output), fit),
            epsilon=final_norm.epsilon,
            ridge=self.ridge,
        )
        embedding_install = _blend_covered_rows(
            target_embedding, embedding_rows, embedding_mask, role
        )
        head_install = _blend_covered_rows(target_head, head_rows, head_mask, role)
        covered = int(embedding_mask.sum().item())
        report = {
            "schema": INTERFACE_SCHEMA,
            "kind": "chart_projected_independent_vocabulary_pair_code_interface",
            "method_id": self.method_id,
            "cross_width_schema": CROSS_WIDTH_BRANCH_FOLD_SCHEMA,
            "cross_width_residual_crossing": True,
            "cross_family_token_interface": True,
            "parent_residual_width": chart_set.source_width,
            "child_residual_width": chart_set.target_width,
            "boundary_chart_set_sha256": chart_set.set_sha256,
            "boundary_chart_policy_sha256": chart_set.report["policy_sha256"],
            "embedding_boundary_chart_sha256": embedding_chart.chart_sha256,
            "terminal_boundary_chart_sha256": terminal_chart.chart_sha256,
            "embedding_boundary_interface_locked": embedding_chart.interface_locked,
            "terminal_boundary_interface_locked": terminal_chart.interface_locked,
            "bridge_kind": artifact.bridge_kind,
            "bridge_artifact_sha256": artifact.artifact_sha256,
            "pair_profile_fingerprint": artifact.profile_fingerprint,
            "parent_tokenizer_semantics_sha256": (
                artifact.parent_tokenizer_semantics_sha256
            ),
            "child_tokenizer_semantics_sha256": (
                artifact.child_tokenizer_semantics_sha256
            ),
            "embedding_weighting": "anchor-preferred",
            "lm_head_weighting": "anchor-preferred",
            "embedding_transport_report_sha256": embedding_reports,
            "lm_head_transport_report_sha256": head_reports,
            "embedding_coordinate_sha256": embedding_chart.map_sha256,
            "lm_head_coordinate_sha256": terminal_chart.map_sha256,
            "cached_parent_evidence_sha256": sealed.cache_key,
            "final_norm_report_sha256": norm_fold.report["report_sha256"],
            "final_norm_relative_rmse": norm_fold.report["relative_rmse"],
            "source_row_capacity": profile.source_row_capacity,
            "source_vocabulary_size": profile.source_vocab_size,
            "target_row_capacity": profile.target_row_capacity,
            "target_vocabulary_size": profile.target_vocab_size,
            "covered_output_rows": covered,
            "uncovered_output_rows": profile.target_row_capacity - covered,
            "masked_row_policy": MASKED_ROW_POLICY,
            "installed_parameter_paths": list(paths),
            "target_tied_storage": tied,
            "parent_forward_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "random_target_rows": 0,
            "zero_gradient": True,
            **_uncovered_record("embedding", embedding_mask),
            **_uncovered_record("lm_head", head_mask),
        }
        incumbent = context.child_state_fingerprint
        blank_rows = (
            (
                target_embedding_path,
                target_embedding.detach().clone(),
                embedding_mask,
            ),
            (target_head_path, target_head.detach().clone(), head_mask),
        )
        snapshot = capture_model_state(
            tuple((path, context.child_model.get_parameter(path)) for path in paths),
            self.snapshot_config,
        )
        try:
            with torch.no_grad():
                for path, value in (
                    (target_embedding_path, embedding_install),
                    (target_head_path, head_install),
                    (final_norm_path, norm_fold.weight),
                ):
                    parameter = context.child_model.get_parameter(path)
                    if tuple(parameter.shape) != tuple(value.shape):
                        raise DynamicCausalOperatorPromotionError(
                            f"{role} compiled {path!r} at the wrong shape."
                        )
                    parameter.copy_(value.to(dtype=parameter.dtype))
            for path, blank, mask in blank_rows:
                live = context.child_model.get_parameter(path).detach()
                if not bool(torch.equal(live[~mask], blank[~mask])):
                    raise DynamicCausalOperatorPromotionError(
                        f"{role} overwrote a target row the pair bridge does "
                        f"not cover in {path!r}."
                    )
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(
                    {path: context.child_model.get_parameter(path) for path in paths}
                )
            finally:
                snapshot.cleanup()
            raise
        return TokenInterfaceBridgeTransaction(
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


class AcceptStructuralCandidate:
    """Behavior stays the outer authority; structure is admitted here.

    The behavioral evidence this run produces is the decoded continuation
    corpus at the end, read by a human.  Nothing inside the executor is allowed
    to substitute a scalar for that.
    """

    def evaluate(self, context: Any) -> FoldDecision:
        del context
        return FoldDecision.accept("Structurally complete cross-family branch.")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _build_parent(
    *,
    geometry: Mapping[str, int],
    vocabulary_size: int,
    layers: int,
    seed: int,
) -> Any:
    torch.manual_seed(seed)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=vocabulary_size,
            hidden_size=geometry["hidden_size"],
            intermediate_size=geometry["intermediate_size"],
            num_hidden_layers=layers,
            num_attention_heads=geometry["num_attention_heads"],
            num_key_value_heads=geometry["num_key_value_heads"],
            head_dim=geometry["head_dim"],
            max_position_embeddings=geometry["max_position_embeddings"],
            rms_norm_eps=geometry["rms_norm_eps_micro"] * 1e-6,
            rope_theta=geometry["rope_theta_kilo"] * 1000.0,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            pad_token_id=0,
        )
    )
    model.eval()
    return model


def confine_parent_residual(model: Any, rank: int, seed: int) -> dict[str, Any]:
    """Confine every residual state of the parent to an exact ``rank`` subspace.

    This is the stage-1 exact-subspace control
    ``tests/test_cross_width_causal_fold_execution.py`` uses, applied at real
    geometry: every residual-facing weight is pushed into one ``rank``
    dimensional subspace and every RMSNorm gain is set to one, so the parent's
    residual states lie exactly in ``rank`` dimensions and a chart that finds
    that subspace discards nothing.  It makes the residual crossing exact on
    purpose, which is the only way to read the branch numbers as branch numbers.
    """

    generator = torch.Generator().manual_seed(seed)
    width = int(model.config.hidden_size)
    basis, _ = torch.linalg.qr(torch.randn(width, rank, generator=generator))
    with torch.no_grad():

        def consume(path: str) -> None:
            weight = model.get_parameter(path)
            weight.copy_((weight @ basis) @ basis.T)

        def produce(path: str) -> None:
            weight = model.get_parameter(path)
            weight.copy_(basis @ (basis.T @ weight))

        consume("model.embed_tokens.weight")
        consume("lm_head.weight")
        model.get_parameter("model.norm.weight").fill_(1.0)
        for index in range(int(model.config.num_hidden_layers)):
            prefix = f"model.layers.{index}."
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
            ):
                consume(prefix + name + ".weight")
            for name in ("self_attn.o_proj", "mlp.down_proj"):
                produce(prefix + name + ".weight")
            model.get_parameter(prefix + "input_layernorm.weight").fill_(1.0)
            model.get_parameter(
                prefix + "post_attention_layernorm.weight"
            ).fill_(1.0)
    return {
        "applied": True,
        "rank": int(rank),
        "seed": int(seed),
        "basis_sha256": canonical_sha256(
            {
                "schema": RUNNER_SCHEMA,
                "kind": "parent_residual_confinement_basis",
                "shape": [int(size) for size in basis.shape],
                "bytes": basis.contiguous().view(torch.uint8).numpy().tobytes().hex(),
            }
        ),
    }


def _build_child(
    *,
    parent_config: Any,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    vocabulary_size: int,
    seed: int,
) -> Any:
    """One blank qwen3-shaped child.

    The blank state is deliberately not zero: a pair bridge leaves rows it
    cannot reach untouched and the transport builder returns exact zeros for
    exactly those rows, so a zero blank would make "left alone" and "installed
    the builder's zeros" indistinguishable.  The rotary contract is copied from
    the parent because the planner compares the position node attribute for
    attribute; ``head_dim`` in particular must match or the crossing is gapped.
    """

    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=vocabulary_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=int(parent_config.num_hidden_layers),
            num_attention_heads=query_heads,
            num_key_value_heads=key_value_heads,
            head_dim=int(parent_config.head_dim),
            max_position_embeddings=int(parent_config.max_position_embeddings),
            rms_norm_eps=float(parent_config.rms_norm_eps),
            rope_theta=float(dict(parent_config.rope_parameters)["rope_theta"]),
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            pad_token_id=0,
        )
    )
    model.eval()
    return model


def _parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _load_prompts(path: Path) -> tuple[str, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("prompts") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise RealScaleFoldError(
            f"{path} must hold a nonempty JSON list of prompt strings."
        )
    prompts = []
    for row in rows:
        value = row.get("prompt") if isinstance(row, dict) else row
        if not isinstance(value, str) or not value.strip():
            raise RealScaleFoldError("Every calibration prompt must be nonempty text.")
        prompts.append(value)
    return tuple(prompts)


def _child_ids_for(
    tokenizer: Any,
    prompts: Sequence[str],
    window: int,
    *,
    pad: bool,
) -> tuple[Any, Any, tuple[int, ...]]:
    """Tokenize with the *child* tokenizer into one fixed-width batch.

    The parent's evidence rows and the child's recaptured rows have to describe
    the same token at the same position, so the tokenization that defines the
    positions has to be the narrower one.  The parent then reads the donor
    preimage of exactly these rows.

    Short prompts are padded on the *right*.  Under causal attention a real
    token never attends to a later position, so right padding leaves every real
    row identical to what an unpadded forward would produce, and the padded rows
    are excluded from the evidence by the attention mask anyway.
    """

    rows: list[list[int]] = []
    masks: list[list[int]] = []
    lengths: list[int] = []
    for prompt in prompts:
        ids = [
            int(value)
            for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ][:window]
        if len(ids) < window and not pad:
            raise RealScaleFoldError(
                f"A prompt tokenizes to {len(ids)} target tokens, below the "
                f"requested unpadded window of {window}."
            )
        missing = window - len(ids)
        lengths.append(len(ids))
        rows.append(ids + [0] * missing)
        masks.append([1] * len(ids) + [0] * missing)
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
        tuple(lengths),
    )


def _donor_preimage(child_ids: Any, donor_index: Any) -> Any:
    return donor_index.index_select(0, child_ids.reshape(-1)).reshape(child_ids.shape)


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------


def _greedy_continuation(
    model: Any,
    prompt_ids: Sequence[int],
    *,
    new_tokens: int,
    restrict: Any | None = None,
) -> list[int]:
    """Greedy decode with a KV cache; no sampling, no temperature, no beams."""

    ids = torch.tensor([list(prompt_ids)], dtype=torch.long)
    produced: list[int] = []
    with torch.no_grad():
        output = model(input_ids=ids, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        for _ in range(new_tokens):
            if restrict is None:
                token = int(logits.argmax(dim=-1).item())
            else:
                local = int(logits.index_select(1, restrict).argmax(dim=-1).item())
                token = int(restrict[local].item())
            produced.append(token)
            step = torch.tensor([[token]], dtype=torch.long)
            output = model(input_ids=step, past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[:, -1, :]
    return produced


def _relative_rmse(prediction: Any, target: Any) -> float:
    difference = (prediction - target).double()
    reference = target.double()
    denominator = float(reference.square().mean().sqrt().item())
    if denominator <= 0.0:
        raise RealScaleFoldError("The comparison target has no energy.")
    return float(difference.square().mean().sqrt().item()) / denominator


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _configuration_stage(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Seal every input that can change a downstream number."""

    geometry = dict(GEOMETRIES[args.parent_geometry])
    prompts_path = args.calibration_prompts.expanduser().resolve(strict=True)
    prompts = _load_prompts(prompts_path)
    calibration = prompts[: args.calibration_prompts_count]
    holdout = prompts[
        len(prompts) - args.holdout_prompts_count : len(prompts)
    ]
    if len(calibration) != args.calibration_prompts_count or len(holdout) != (
        args.holdout_prompts_count
    ):
        raise RealScaleFoldError(
            "The prompt corpus is too small for the requested calibration and "
            "held-out split."
        )
    if set(calibration) & set(holdout):
        raise RealScaleFoldError(
            "Calibration and held-out prompts overlap; the decoded evidence "
            "would not be held out."
        )
    layers = args.layers if args.layers is not None else geometry["num_hidden_layers"]
    if layers < 1 or layers > geometry["num_hidden_layers"]:
        raise RealScaleFoldError(
            f"--layers must fall in 1..{geometry['num_hidden_layers']}."
        )
    identity = {
        "schema": RUNNER_SCHEMA,
        "stage": "configuration",
        "parent": {
            "geometry_name": args.parent_geometry,
            "geometry": geometry,
            "folded_layer_prefix": layers,
            "full_depth": geometry["num_hidden_layers"],
            "bounded_layer_prefix": layers < geometry["num_hidden_layers"],
            "weight_source": args.parent_weights,
            "pretrained": args.parent_weights != "seeded-random-init",
            "seed": args.parent_seed,
            "residual_confinement_rank": args.parent_residual_rank,
        },
        "child": {
            "family": "qwen3",
            "hidden_size": args.child_hidden_size,
            "intermediate_size": args.child_intermediate_size,
            "num_attention_heads": args.child_query_heads,
            "num_key_value_heads": args.child_key_value_heads,
            "vocab_size": args.child_vocab_size,
            "seed": args.child_seed,
            "tie_word_embeddings": False,
            "query_key_norm": "per_head_rms_norm",
        },
        "tokenizers": {
            "source_repository": args.source_tokenizer,
            "source_local_files_only": True,
            "target_construction": "prune_hf_tokenizer_subset_of_source",
            "target_vocab_size": args.child_vocab_size,
            "bridge_kind": "sealed-independent-tokenizer-pair-bridge",
        },
        "corpus": {
            "path": str(prompts_path),
            "file_sha256": _file_sha256(prompts_path),
            "total_prompts": len(prompts),
            "calibration_prompt_indices": list(range(len(calibration))),
            "holdout_prompt_indices": list(
                range(len(prompts) - len(holdout), len(prompts))
            ),
            "calibration_prompt_count": len(calibration),
            "holdout_prompt_count": len(holdout),
            "evidence_window_tokens": args.evidence_window,
            "holdout_prompt_window_tokens": args.holdout_window,
            "decode_new_tokens": args.decode_new_tokens,
            "calibration_sha256": canonical_sha256(
                {"kind": "calibration", "prompts": list(calibration)}
            ),
            "holdout_sha256": canonical_sha256(
                {"kind": "holdout", "prompts": list(holdout)}
            ),
        },
        "fold": {
            "enabled_experimental_method_ids": sorted(CAUSAL_LAYER_BRANCH_METHODS),
            "ridge": args.ridge,
            "snapshot_backend": "memory",
        },
    }
    # ``input_sha256`` is the run's identity and every later stage chains off
    # it.  A wall clock, a thread count and a library version are recorded but
    # deliberately excluded from it: none of them changes a number a downstream
    # stage computes, and folding them in would make the identity differ on
    # every rerun, which would silently turn a resumable driver into one that
    # recomputes the whole fold each time it is restarted.
    body = {
        **identity,
        "input_sha256": canonical_sha256(identity),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "torch_threads": int(torch.get_num_threads()),
        "device": "cpu",
    }
    receipt = _write_receipt(root / "configuration.json", body)
    return {
        "receipt": receipt,
        "calibration": calibration,
        "holdout": holdout,
        "layers": layers,
        "geometry": geometry,
    }


def _tokenizer_stage(
    args: argparse.Namespace,
    root: Path,
    configuration: Mapping[str, Any],
    calibration: Sequence[str],
) -> dict[str, Any]:
    """Compile the real target tokenizer and the sealed pair bridge."""

    source = transformers.AutoTokenizer.from_pretrained(
        args.source_tokenizer,
        local_files_only=True,
    )
    target_directory = root / "target-tokenizer"
    bridge_directory = root / "pair-bridge"
    receipt_path = root / "tokenizers.json"
    input_sha256 = canonical_sha256(
        {
            "configuration": configuration["input_sha256"],
            "source_tokenizer_semantics_sha256": (
                fingerprint_hf_tokenizer_semantics(source)
            ),
        }
    )
    existing = _read_receipt(receipt_path)
    if (
        existing is not None
        and existing.get("input_sha256") == input_sha256
        and target_directory.is_dir()
        and bridge_directory.is_dir()
    ):
        target = transformers.AutoTokenizer.from_pretrained(
            target_directory,
            local_files_only=True,
        )
        bridge = compile_hf_tokenizer_pair_bridge(source, target, bridge_directory)
        donor = torch.tensor(existing["donor_row_indices"], dtype=torch.long)
        return {
            "receipt": existing,
            "source": source,
            "target": target,
            "bridge": bridge,
            "donor_index": donor,
            "resumed": True,
        }
    pruned = prune_hf_tokenizer(
        source,
        target_directory,
        target_vocab_size=args.child_vocab_size,
        calibration_texts=tuple(calibration),
    )
    target = pruned.tokenizer
    bridge = compile_hf_tokenizer_pair_bridge(source, target, bridge_directory)
    donor = torch.tensor(pruned.selected_donor_token_ids, dtype=torch.long)
    body = {
        "schema": RUNNER_SCHEMA,
        "stage": "tokenizers",
        "input_sha256": input_sha256,
        "source_repository": args.source_tokenizer,
        "source_class": type(source).__name__,
        "source_vocab_size": int(len(source)),
        "source_tokenizer_semantics_sha256": fingerprint_hf_tokenizer_semantics(source),
        "target_directory": target_directory.name,
        "target_class": type(target).__name__,
        "target_vocab_size": int(len(target)),
        "target_tokenizer_semantics_sha256": fingerprint_hf_tokenizer_semantics(target),
        "target_construction": (
            "prune_hf_tokenizer: real ByteLevel-BPE subset of the source, with "
            "complete byte coverage, every special token and complete BPE merge "
            "dependency chains, selected on the calibration corpus"
        ),
        "retained_merges": len(pruned.retained_merges),
        "bridge_directory": bridge_directory.name,
        "bridge_kind": bridge.bridge_kind,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "pair_profile_fingerprint": bridge.profile_fingerprint,
        "bridge_source_row_capacity": bridge.profile.source_row_capacity,
        "bridge_target_row_capacity": bridge.profile.target_row_capacity,
        "donor_row_indices": [int(value) for value in pruned.selected_donor_token_ids],
        "donor_row_index_sha256": canonical_sha256(
            {
                "kind": "evaluation_alignment_donor_rows",
                "rows": [int(value) for value in pruned.selected_donor_token_ids],
            }
        ),
        "donor_row_index_role": (
            "evaluation alignment only; the fold installs vocabulary rows "
            "through the sealed pair bridge, never through this index"
        ),
    }
    receipt = _write_receipt(receipt_path, body)
    return {
        "receipt": receipt,
        "source": source,
        "target": target,
        "bridge": bridge,
        "donor_index": donor,
        "resumed": False,
    }


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
    }


def _chart_summary(chart_set: ResidualBoundaryChartSet) -> dict[str, Any]:
    cycle: list[float] = []
    fluctuation: list[float] = []
    fit_retained: list[float] = []
    heldout_retained: list[float] = []
    unabsorbed: list[float] = []
    for chart in chart_set.charts:
        diagnostics = dict(chart.report["diagnostics"])
        cycle.append(float(diagnostics["cycle_defect"]))
        fluctuation.append(float(diagnostics["norm_commutation_row_fluctuation"]))
        absorption = diagnostics.get("absorption_demand")
        if absorption is not None:
            unabsorbed.append(float(absorption["unabsorbed_energy_fraction"]))
        chosen = dict(chart.report["lineage"].get("selection", {}))
        for name, sink in (
            ("fit_retained_energy_fraction", fit_retained),
            ("heldout_retained_energy_fraction", heldout_retained),
        ):
            value = chosen.get(name)
            if isinstance(value, (int, float)):
                sink.append(float(value))
    return {
        "boundary_count": len(chart_set.charts),
        "interface_locked_boundary_ids": [
            chart.boundary_id for chart in chart_set.charts if chart.interface_locked
        ],
        "cycle_defect": _summarize(cycle),
        "norm_commutation_row_fluctuation": _summarize(fluctuation),
        "fit_retained_energy_fraction": _summarize(fit_retained),
        "heldout_retained_energy_fraction": _summarize(heldout_retained),
        "absorption_demand_unabsorbed_fraction": _summarize(unabsorbed),
    }


def _fold_stage(
    args: argparse.Namespace,
    root: Path,
    configuration: Mapping[str, Any],
    tokenizers: Mapping[str, Any],
    calibration: Sequence[str],
    layers: int,
    geometry: Mapping[str, int],
) -> dict[str, Any]:
    """One parent forward, one chart set, one executor run, one folded child."""

    receipt_path = root / "fold.json"
    child_directory = root / "child"
    input_sha256 = canonical_sha256(
        {
            "configuration": configuration["input_sha256"],
            "tokenizers": tokenizers["receipt"]["receipt_sha256"],
        }
    )
    existing = _read_receipt(receipt_path)
    if (
        existing is not None
        and existing.get("input_sha256") == input_sha256
        and (child_directory / "config.json").is_file()
    ):
        child = transformers.Qwen3ForCausalLM.from_pretrained(
            child_directory,
            local_files_only=True,
        )
        child.eval()
        if fingerprint_model_state(child) != existing["folded_child_state_sha256"]:
            raise RealScaleFoldError(
                "The saved folded child does not match its sealed fingerprint."
            )
        return {"receipt": existing, "child": child, "resumed": True}

    source_tokenizer = tokenizers["source"]
    target_tokenizer = tokenizers["target"]
    donor_index = tokenizers["donor_index"]
    started = time.perf_counter()

    parent = _build_parent(
        geometry=geometry,
        vocabulary_size=int(len(source_tokenizer)),
        layers=layers,
        seed=args.parent_seed,
    )
    confinement: dict[str, Any] = {"applied": False}
    if args.parent_residual_rank is not None:
        confinement = confine_parent_residual(
            parent,
            args.parent_residual_rank,
            args.parent_seed + 1,
        )
    child = _build_child(
        parent_config=parent.config,
        hidden_size=args.child_hidden_size,
        intermediate_size=args.child_intermediate_size,
        query_heads=args.child_query_heads,
        key_value_heads=args.child_key_value_heads,
        vocabulary_size=int(len(target_tokenizer)),
        seed=args.child_seed,
    )
    blank_state_fingerprint = fingerprint_model_state(child)

    child_ids, attention_mask, lengths = _child_ids_for(
        target_tokenizer,
        calibration,
        args.evidence_window,
        pad=True,
    )
    parent_ids = _donor_preimage(child_ids, donor_index)

    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(source_tokenizer),
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(target_tokenizer),
    )
    capture_started = time.perf_counter()
    evidence = capture_hf_causal_circuit_evidence(
        parent,
        {"input_ids": parent_ids, "attention_mask": attention_mask},
        parent_graph_fingerprint=parent_graph.fingerprint,
        metadata={
            "runner": RUNNER_SCHEMA,
            "corpus": "kernel-stage2 calibration prompts, donor preimage rows",
        },
    )
    capture_seconds = time.perf_counter() - capture_started
    evidence_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "evidence",
        "input_sha256": input_sha256,
        "cached_parent_evidence_sha256": evidence.cache_key,
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "parent_forward_calls": int(evidence.parent_forward_calls),
        "sequences": int(parent_ids.shape[0]),
        "sequence_length": int(parent_ids.shape[1]),
        "unpadded_token_lengths": list(lengths),
        "padded_positions": int(parent_ids.numel() - sum(lengths)),
        "valid_rows": int(evidence.valid_token_mask.sum().item()),
        "fit_rows": int(evidence.fit_token_mask.sum().item()),
        "validation_rows": int(evidence.validation_token_mask.sum().item()),
        "layer_count": len(evidence.layers),
        "capture_seconds": capture_seconds,
        "child_recapture_input_ids_sha256": canonical_sha256(
            {
                "kind": "child_recapture_input_ids",
                "shape": [int(size) for size in child_ids.shape],
                "ids": [int(value) for value in child_ids.reshape(-1).tolist()],
            }
        ),
    }
    _write_receipt(root / "evidence.json", evidence_body)

    chart_started = time.perf_counter()
    chart_set = compile_residual_boundary_chart_set(evidence, parent, child)
    chart_seconds = time.perf_counter() - chart_started
    chart_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "charts",
        "input_sha256": input_sha256,
        "cached_parent_evidence_sha256": evidence.cache_key,
        "boundary_chart_set_sha256": chart_set.set_sha256,
        "boundary_chart_policy_sha256": chart_set.report["policy_sha256"],
        "source_width": chart_set.source_width,
        "target_width": chart_set.target_width,
        "boundary_ids": list(chart_set.boundary_ids),
        "compile_seconds": chart_seconds,
        "summary": _chart_summary(chart_set),
    }
    _write_receipt(root / "charts.json", chart_body)

    plan = DynamicFoldPlanner(
        enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS
    ).plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=blank_state_fingerprint,
        evidence_fingerprint=evidence.cache_key,
        parent_state_fingerprints={"parent": fingerprint_model_state(parent)},
    )
    if not plan.ready or plan.gaps:
        raise RealScaleFoldError(
            "The cross-family cross-width plan is not gap free: "
            + ", ".join(sorted({gap.required_capability for gap in plan.gaps}))
        )
    recapture = LiveChildBoundaryRecapture(
        {"input_ids": child_ids, "attention_mask": attention_mask}
    )
    registry = FoldOperatorRegistry(
        (
            ExactCopyOperator(),
            CrossWidthCausalBranchOperator(
                CausalGQAAttentionBranchOperator(ridge=args.ridge),
                chart_set=chart_set,
                recapture=recapture,
            ),
            CrossWidthCausalBranchOperator(
                CausalGatedMLPBranchOperator(ridge=args.ridge),
                chart_set=chart_set,
                recapture=recapture,
            ),
            ChartProjectedPairCodeTokenHeadInterfaceOperator(
                artifact=tokenizers["bridge"],
                chart_set=chart_set,
                ridge=args.ridge,
            ),
        )
    )
    context = FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=child,
        parent_graphs={"parent": parent_graph},
        child_graph=child_graph,
    )
    parent_forwards = 0

    def counted(_module: Any, _args: Any, _output: Any) -> None:
        nonlocal parent_forwards
        parent_forwards += 1

    handle = parent.register_forward_hook(counted)
    execute_started = time.perf_counter()
    try:
        run = DynamicFoldExecutor(
            operators=registry,
            evidence_provider=StaticEvidenceProvider(evidence.cache_key, evidence),
            evaluator=AcceptStructuralCandidate(),
        ).execute(plan, context)
    except DynamicCausalOperatorPromotionError as exc:
        # A fail-closed branch is a measurement, not a crash: seal where the
        # composition stopped so the depth it reached is on the record, then
        # let the caller fold the surviving prefix instead.
        _write_receipt(
            root / "fold-failure.json",
            {
                "schema": RUNNER_SCHEMA,
                "stage": "fold-failure",
                "input_sha256": input_sha256,
                "requested_layer_prefix": layers,
                "cached_parent_evidence_sha256": evidence.cache_key,
                "boundary_chart_set_sha256": chart_set.set_sha256,
                "child_recapture_calls": recapture.recapture_calls,
                "failure_class": type(exc).__name__,
                "failure": str(exc),
                "child_state_rolled_back_to_blank": (
                    fingerprint_model_state(child) == blank_state_fingerprint
                ),
                "seconds": time.perf_counter() - execute_started,
            },
        )
        raise
    finally:
        handle.remove()
    execute_seconds = time.perf_counter() - execute_started
    if run.status != "complete":
        raise RealScaleFoldError(
            f"The fold run ended {run.status!r} rather than complete."
        )

    branches: list[dict[str, Any]] = []
    interface: dict[str, Any] = {}
    for attempt in run.attempts:
        report = dict(attempt.transaction_report)
        if attempt.method_id == TOKEN_INTERFACE_BRIDGE_METHOD:
            interface = {
                key: report[key]
                for key in (
                    "bridge_kind",
                    "covered_output_rows",
                    "uncovered_output_rows",
                    "target_row_capacity",
                    "source_row_capacity",
                    "masked_row_policy",
                    "final_norm_relative_rmse",
                    "embedding_uncovered_target_rows",
                    "lm_head_uncovered_target_rows",
                    "embedding_evidence_mask_sha256",
                    "lm_head_evidence_mask_sha256",
                    "boundary_chart_set_sha256",
                    "bridge_artifact_sha256",
                )
                if key in report
            }
            continue
        if attempt.method_id not in {
            CAUSAL_ATTENTION_BRANCH_METHOD,
            CAUSAL_GATED_MLP_BRANCH_METHOD,
        }:
            continue
        key = (
            "validation_attention_delta_relative_rmse"
            if attempt.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
            else "validation_mlp_delta_relative_rmse"
        )
        target = dict(report["sequential_branch_target"])
        row = {
            "method_id": attempt.method_id,
            "source_layer_index": report["source_layer_index"],
            "input_boundary_id": report["input_boundary_id"],
            "output_boundary_id": report["output_boundary_id"],
            "validation_endpoint_relative_rmse": report[key],
            "no_branch_endpoint_relative_rmse_baseline": report[
                "no_branch_endpoint_relative_rmse_baseline"
            ],
            "parent_residual_width": report["parent_residual_width"],
            "child_residual_width": report["child_residual_width"],
            "sequential_child_replay": report["sequential_child_replay"],
            "parent_residual_coordinate_endpoints": report[
                "parent_residual_coordinate_endpoints"
            ],
            "parent_forward_calls": report["parent_forward_calls"],
            "reverse_reclosure_sweep_executed": report[
                "reverse_reclosure_sweep_executed"
            ],
            "fit_upstream_error_energy_ratio": target[
                "fit_upstream_error_energy_ratio"
            ],
            "sequential_expansion_max_abs_error": target[
                "sequential_expansion_max_abs_error"
            ],
            "input_boundary_cycle_defect": report["input_boundary_chart"][
                "cycle_defect"
            ],
            "output_boundary_cycle_defect": report["output_boundary_chart"][
                "cycle_defect"
            ],
        }
        if attempt.method_id == CAUSAL_ATTENTION_BRANCH_METHOD:
            row["target_query_key_norm"] = report["target_query_key_norm"]
            row["query_norm_gain_relative_rmse"] = report[
                "query_norm_gain_relative_rmse"
            ]
            row["key_norm_gain_relative_rmse"] = report["key_norm_gain_relative_rmse"]
        branches.append(row)

    child_directory.mkdir(parents=True, exist_ok=True)
    child.save_pretrained(child_directory, safe_serialization=True)
    target_tokenizer.save_pretrained(child_directory)
    folded_fingerprint = fingerprint_model_state(child)
    changed = [
        path
        for attempt in run.attempts
        for path in attempt.transaction_report["changed_state_paths"]
    ]
    body = {
        "schema": RUNNER_SCHEMA,
        "stage": "fold",
        "input_sha256": input_sha256,
        "plan": {
            "ready": bool(plan.ready),
            "gap_count": len(plan.gaps),
            "action_count": len(plan.actions),
            "method_counts": {
                method: sum(
                    1 for action in plan.actions if action.method_id == method
                )
                for method in sorted({action.method_id for action in plan.actions})
            },
            "plan_fingerprint": plan.fingerprint,
            "partitions_every_learned_child_tensor": (
                sorted(
                    path
                    for action in plan.actions
                    for path in action.target_parameter_paths
                )
                == sorted(child_graph.parameter_paths)
            ),
        },
        "run": {
            "status": run.status,
            "fingerprint": run.fingerprint,
            "attempts": len(run.attempts),
            "accepted": sum(1 for attempt in run.attempts if attempt.accepted),
            "observed_parent_root_forward_calls": (
                run.observed_parent_root_forward_calls
            ),
            "observed_parent_module_forward_calls": (
                run.observed_parent_module_forward_calls
            ),
            "hooked_parent_forward_calls_during_execution": parent_forwards,
            "parent_forward_calls_during_evidence_capture": int(
                evidence.parent_forward_calls
            ),
            "optimizer_constructions": run.optimizer_constructions,
            "optimizer_steps": run.optimizer_steps,
            "backward_calls": run.backward_calls,
            "autograd_grad_calls": run.autograd_grad_calls,
            "pruning_operations": run.pruning_operations,
            "quantization_operations": run.quantization_operations,
            "executed_fold_static_zero_training_audit_passed": (
                run.executed_fold_static_zero_training_audit_passed
            ),
            "operator_static_zero_training_audit_passed": (
                run.operator_static_zero_training_audit_passed
            ),
            "child_recapture_calls": recapture.recapture_calls,
            "changed_state_paths": len(changed),
            "changed_state_paths_unique": len(changed) == len(set(changed)),
            "completed_every_learned_child_tensor": (
                set(run.completed_target_parameter_paths)
                == set(child_graph.parameter_paths)
            ),
        },
        "parent": {
            "family": "llama",
            "hidden_size": int(parent.config.hidden_size),
            "layers": int(parent.config.num_hidden_layers),
            "parameters": _parameter_count(parent),
            "vocab_size": int(parent.config.vocab_size),
            "state_sha256": fingerprint_model_state(parent),
            "residual_confinement": confinement,
        },
        "child": {
            "family": "qwen3",
            "hidden_size": int(child.config.hidden_size),
            "layers": int(child.config.num_hidden_layers),
            "parameters": _parameter_count(child),
            "vocab_size": int(child.config.vocab_size),
            "blank_state_sha256": blank_state_fingerprint,
            "directory": child_directory.name,
        },
        "branches": branches,
        "branch_summary": {
            "attention_validation_endpoint_relative_rmse": _summarize(
                [
                    row["validation_endpoint_relative_rmse"]
                    for row in branches
                    if row["method_id"] == CAUSAL_ATTENTION_BRANCH_METHOD
                ]
            ),
            "gated_mlp_validation_endpoint_relative_rmse": _summarize(
                [
                    row["validation_endpoint_relative_rmse"]
                    for row in branches
                    if row["method_id"] == CAUSAL_GATED_MLP_BRANCH_METHOD
                ]
            ),
        },
        "token_interface": interface,
        "folded_child_state_sha256": folded_fingerprint,
        "child_state_changed_by_fold": folded_fingerprint != blank_state_fingerprint,
        "seconds": {
            "evidence_capture": capture_seconds,
            "chart_compile": chart_seconds,
            "execute": execute_seconds,
            "stage_total": time.perf_counter() - started,
        },
    }
    receipt = _write_receipt(receipt_path, body)
    return {"receipt": receipt, "child": child, "resumed": False}


def _behavior_stage(
    args: argparse.Namespace,
    root: Path,
    configuration: Mapping[str, Any],
    tokenizers: Mapping[str, Any],
    fold: Mapping[str, Any],
    holdout: Sequence[str],
    layers: int,
    geometry: Mapping[str, int],
) -> dict[str, Any]:
    """Teacher-forced logit agreement, then complete decoded continuations.

    Both models read the same text at the same positions: the held-out prompts
    are tokenized by the *child* tokenizer and the parent reads the donor
    preimage of exactly those rows.  The parent's logits are then restricted to
    the child's row set, which is the only comparison between a 50k-row head and
    a reduced one that is not an alignment artifact.
    """

    receipt_path = root / "behavior.json"
    input_sha256 = canonical_sha256(
        {
            "configuration": configuration["input_sha256"],
            "tokenizers": tokenizers["receipt"]["receipt_sha256"],
            "fold": fold["receipt"]["receipt_sha256"],
            "decode_new_tokens": args.decode_new_tokens,
            "holdout_window": args.holdout_window,
        }
    )
    existing = _read_receipt(receipt_path)
    if existing is not None and existing.get("input_sha256") == input_sha256:
        return {"receipt": existing, "resumed": True}

    started = time.perf_counter()
    source_tokenizer = tokenizers["source"]
    target_tokenizer = tokenizers["target"]
    donor_index = tokenizers["donor_index"]
    child = fold["child"]
    parent = _build_parent(
        geometry=geometry,
        vocabulary_size=int(len(source_tokenizer)),
        layers=layers,
        seed=args.parent_seed,
    )
    if args.parent_residual_rank is not None:
        confine_parent_residual(parent, args.parent_residual_rank, args.parent_seed + 1)
    if fingerprint_model_state(parent) != fold["receipt"]["parent"]["state_sha256"]:
        raise RealScaleFoldError(
            "The rebuilt parent does not match the parent the fold read."
        )

    child_ids, _mask, lengths = _child_ids_for(
        target_tokenizer,
        holdout,
        args.holdout_window,
        pad=False,
    )
    parent_ids = _donor_preimage(child_ids, donor_index)
    with torch.no_grad():
        child_logits = child(input_ids=child_ids, use_cache=False).logits
        parent_logits = parent(input_ids=parent_ids, use_cache=False).logits
    aligned_parent = parent_logits.index_select(2, donor_index)
    logit_rmse = _relative_rmse(child_logits, aligned_parent)
    centered_rmse = _relative_rmse(
        child_logits - child_logits.mean(dim=-1, keepdim=True),
        aligned_parent - aligned_parent.mean(dim=-1, keepdim=True),
    )
    child_top1 = child_logits.argmax(dim=-1)
    parent_top1 = aligned_parent.argmax(dim=-1)
    top1_agreement = float((child_top1 == parent_top1).double().mean().item())
    del lengths

    continuations: list[dict[str, Any]] = []
    for index, prompt in enumerate(holdout):
        prompt_child_ids = [int(value) for value in child_ids[index].tolist()]
        prompt_parent_ids = [int(value) for value in parent_ids[index].tolist()]
        child_tokens = _greedy_continuation(
            child,
            prompt_child_ids,
            new_tokens=args.decode_new_tokens,
        )
        parent_tokens = _greedy_continuation(
            parent,
            prompt_parent_ids,
            new_tokens=args.decode_new_tokens,
            restrict=donor_index,
        )
        parent_free_tokens = _greedy_continuation(
            parent,
            prompt_parent_ids,
            new_tokens=args.decode_new_tokens,
        )
        matched = 0
        for left, right in zip(child_tokens, parent_tokens):
            if int(donor_index[left].item()) != right:
                break
            matched += 1
        continuations.append(
            {
                "holdout_index": index,
                "prompt_window_text": target_tokenizer.decode(prompt_child_ids),
                "prompt_window_tokens": len(prompt_child_ids),
                "child_continuation_text": target_tokenizer.decode(child_tokens),
                "child_continuation_token_ids": child_tokens,
                "child_continuation_tokens": target_tokenizer.convert_ids_to_tokens(
                    child_tokens
                ),
                "parent_row_restricted_continuation_text": source_tokenizer.decode(
                    parent_tokens
                ),
                "parent_row_restricted_continuation_token_ids": parent_tokens,
                "parent_unrestricted_continuation_text": source_tokenizer.decode(
                    parent_free_tokens
                ),
                "parent_unrestricted_continuation_token_ids": parent_free_tokens,
                "child_parent_greedy_prefix_agreement_tokens": matched,
            }
        )

    body = {
        "schema": RUNNER_SCHEMA,
        "stage": "behavior",
        "input_sha256": input_sha256,
        "holdout_prompt_count": len(holdout),
        "holdout_window_tokens": args.holdout_window,
        "decode_new_tokens": args.decode_new_tokens,
        "decode_policy": "greedy argmax, no sampling, no temperature, no beams",
        "teacher_forced": {
            "rows": int(child_ids.shape[0] * child_ids.shape[1]),
            "child_vs_row_aligned_parent_logit_relative_rmse": logit_rmse,
            "child_vs_row_aligned_parent_centered_logit_relative_rmse": centered_rmse,
            "child_vs_row_aligned_parent_top1_agreement": top1_agreement,
            "alignment": (
                "held-out text tokenized by the child tokenizer; the parent "
                "reads the donor preimage of the same rows and its logits are "
                "restricted to the child's row set"
            ),
        },
        "greedy_prefix_agreement_tokens": _summarize(
            [
                float(row["child_parent_greedy_prefix_agreement_tokens"])
                for row in continuations
            ]
        ),
        "continuations": continuations,
        "seconds": time.perf_counter() - started,
    }
    receipt = _write_receipt(receipt_path, body)
    return {"receipt": receipt, "resumed": False}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_report(root: Path) -> dict[str, Any]:
    """Re-verify every receipt seal and every content digest in the manifest."""

    manifest_path = root / "manifest.json"
    manifest = _read_receipt(manifest_path)
    if manifest is None:
        raise RealScaleFoldError(f"No sealed manifest at {manifest_path}.")
    problems: list[str] = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            problems.append(f"missing: {row['path']}")
            continue
        if _file_sha256(path) != row["sha256"]:
            problems.append(f"digest changed: {row['path']}")
    for stage in STAGES:
        path = root / f"{stage}.json"
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if not verify_receipt(document):
            problems.append(f"receipt seal broken: {stage}.json")
    return {
        "root": str(root),
        "manifest_sha256": manifest["receipt_sha256"],
        "files": len(manifest["files"]),
        "problems": problems,
        "verified": not problems,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.parent_seed)
    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    root = args.report_root.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)

    configuration = _configuration_stage(args, root)
    tokenizers = _tokenizer_stage(
        args,
        root,
        configuration["receipt"],
        configuration["calibration"],
    )
    fold = _fold_stage(
        args,
        root,
        configuration["receipt"],
        tokenizers,
        configuration["calibration"],
        configuration["layers"],
        configuration["geometry"],
    )
    behavior = _behavior_stage(
        args,
        root,
        configuration["receipt"],
        tokenizers,
        fold,
        configuration["holdout"],
        configuration["layers"],
        configuration["geometry"],
    )
    manifest_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "manifest",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {
            "configuration": configuration["receipt"]["receipt_sha256"],
            "tokenizers": tokenizers["receipt"]["receipt_sha256"],
            "fold": fold["receipt"]["receipt_sha256"],
            "behavior": behavior["receipt"]["receipt_sha256"],
        },
        "resumed_stages": sorted(
            name
            for name, stage in (
                ("tokenizers", tokenizers),
                ("fold", fold),
                ("behavior", behavior),
            )
            if stage.get("resumed")
        ),
        "files": _tree_manifest(root, skip=("manifest.json",)),
    }
    manifest = _write_receipt(root / "manifest.json", manifest_body)
    return {
        "report_root": str(root),
        "manifest_sha256": manifest["receipt_sha256"],
        "fold_receipt_sha256": fold["receipt"]["receipt_sha256"],
        "behavior_receipt_sha256": behavior["receipt"]["receipt_sha256"],
        "verification": verify_report(root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fold a real-scale llama parent into a blank qwen3 child across a "
            "residual width contraction and an independent vocabulary, then "
            "decode held-out continuations from the folded child."
        )
    )
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--calibration-prompts",
        type=Path,
        default=ROOT / "reports" / "kernel-stage2-20260801"
        / "calibration_prompts_64.json",
    )
    parser.add_argument("--parent-geometry", choices=sorted(GEOMETRIES), default=(
        "smollm2-135m"
    ))
    parser.add_argument(
        "--parent-weights",
        choices=("seeded-random-init",),
        default="seeded-random-init",
        help=(
            "Only the synthesized parent is implemented here.  A pretrained "
            "parent needs its weights present offline; see NOTES-W7A.md."
        ),
    )
    parser.add_argument("--parent-seed", type=int, default=20260803)
    parser.add_argument(
        "--parent-residual-rank",
        type=int,
        default=None,
        help=(
            "Confine the parent's residual states to this exact rank before "
            "capture (the stage-1 exact-subspace control).  Omit for a "
            "full-rank parent."
        ),
    )
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--child-hidden-size", type=int, default=432)
    parser.add_argument("--child-intermediate-size", type=int, default=1152)
    parser.add_argument("--child-query-heads", type=int, default=6)
    parser.add_argument("--child-key-value-heads", type=int, default=3)
    parser.add_argument("--child-vocab-size", type=int, default=16384)
    parser.add_argument("--child-seed", type=int, default=4711)
    parser.add_argument("--source-tokenizer", default="sshleifer/tiny-gpt2")
    parser.add_argument("--calibration-prompts-count", type=int, default=24)
    parser.add_argument("--holdout-prompts-count", type=int, default=16)
    parser.add_argument("--evidence-window", type=int, default=160)
    parser.add_argument("--holdout-window", type=int, default=96)
    parser.add_argument("--decode-new-tokens", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-verify an existing sealed report and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            result = verify_report(args.report_root.expanduser().absolute())
        else:
            result = run(args)
    except (
        RealScaleFoldError,
        DynamicCausalOperatorPromotionError,
        DynamicFoldExecutionError,
        CrossWidthResidualChartError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"real-scale fold failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
