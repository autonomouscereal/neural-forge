"""Exact-bound LFM2 validation for the semantic whole-model ABI.

This adapter authenticates the dense, same-depth, bridge-aware target contract
used by the exact-4x work.  It does not itself wrap that experiment's concrete
compiler sequence: callers must supply exact-bound resources whose compiler
implements the family-neutral ABI.  A configure-once catalog should instead
implement ``SemanticWholeModelResourceResolver``.  The executor, transaction,
evidence, component receipts, publication, and decoded-behavior semantics stay
in :mod:`neural_forge.dynamic_semantic_fold`.
"""

from __future__ import annotations

from .dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
    validate_lfm2_dense_budgeted_child,
)
from .dynamic_semantic_fold import (
    BoundSemanticWholeModelFoldStackAdapter,
    DynamicSemanticFoldError,
    SemanticWholeModelFoldResources,
)
from .dynamic_target_synthesis import DynamicTargetSynthesisResult
from .hf_lfm2_dense_budget_model_fold import DENSE_BUDGET_MODEL_METHOD


LFM2_BOUND_SEMANTIC_WHOLE_MODEL_STACK_ADAPTER_ID = (
    "neural_forge.lfm2.bound_semantic_whole_model"
)


class Lfm2BoundSemanticWholeModelFoldStackAdapter(
    BoundSemanticWholeModelFoldStackAdapter
):
    """Authenticate one exact-bound LFM2 blank before semantic execution.

    ``require_tokenizer_bridge`` and ``require_equal_depth`` describe the
    currently implemented semantic compiler, not permanent facade limits.
    Another LFM2 compiler can register different values without changing the
    generic transaction ABI.
    """

    def __init__(
        self,
        resources: SemanticWholeModelFoldResources,
        *,
        require_tokenizer_bridge: bool = True,
        require_equal_depth: bool = True,
    ) -> None:
        if not isinstance(require_tokenizer_bridge, bool) or not isinstance(
            require_equal_depth,
            bool,
        ):
            raise DynamicSemanticFoldError(
                "LFM2 semantic requirements must be boolean."
            )
        binding = resources.binding
        if (
            binding.target_family != "lfm2"
            or binding.target_modalities != ("text",)
            or binding.target_adapter_id
            != Lfm2TargetSynthesisAdapter.adapter_id
            or binding.target_adapter_version
            != Lfm2TargetSynthesisAdapter.version
            or binding.method_id != DENSE_BUDGET_MODEL_METHOD
            or resources.stack_adapter_id
            != LFM2_BOUND_SEMANTIC_WHOLE_MODEL_STACK_ADAPTER_ID
        ):
            raise DynamicSemanticFoldError(
                "LFM2 semantic resources use a different family, target "
                "adapter, method, or stack identity."
            )
        self.require_tokenizer_bridge = require_tokenizer_bridge
        self.require_equal_depth = require_equal_depth
        super().__init__(resources)

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        if not super().supports(target):
            return False
        selected = target.adapter_payload
        if not isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
            return False
        try:
            validate_lfm2_dense_budgeted_child(
                selected,
                target.model,
                target.graph,
            )
        except Exception:
            return False
        parent_graphs = dict(target.parent_graph_fingerprints)
        if (
            selected.source_parent not in parent_graphs
            or parent_graphs[selected.source_parent]
            != selected.source_graph_fingerprint
            or selected.exact_target_unique_parameters
            != self.resources.binding.exact_target_unique_parameters
            or target.request.tokenizer_fingerprint
            != selected.child_tokenizer_semantics_sha256
        ):
            return False
        if self.require_tokenizer_bridge and (
            not selected.tokenizer_bridge_required
            or selected.tie_word_embeddings
            or selected.target_token_interface_mode != "untied"
        ):
            return False
        if self.require_equal_depth and (
            selected.ordered_source_spans
            != tuple((index,) for index in range(selected.target_depth))
        ):
            return False
        return True


__all__ = [
    "LFM2_BOUND_SEMANTIC_WHOLE_MODEL_STACK_ADAPTER_ID",
    "Lfm2BoundSemanticWholeModelFoldStackAdapter",
]
