"""Deterministic unique-parameter budget IR for tiny GPT-2 and Llama models.

The compiler preserves decoder execution depth.  It reduces unique storage by
factorizing the vocabulary operators and tying deterministic contiguous groups
of MLPs.  Materialization restores dense vocabulary operators and independent
MLP parameter objects without removing any logical layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from neural_forge.hf_arch_layout import (
    DecoderLayout,
    build_decoder_layout,
    resolve_module,
)
from neural_forge.hf_factorized_vocab import (
    FactorizedVocabularyPair,
    VocabularyFactorizationPlan,
    exact_parameter_report,
    factorize_embedding_and_lm_head,
    plan_vocabulary_factorization,
)
from neural_forge.hf_parameter_sharing import (
    MLPModuleLayout,
    expand_mlp_modules,
    extract_mlp_bundle,
    merge_aligned_mlp_bundles,
    parameter_sharing_report,
    tie_mlp_modules,
)


PLAN_SCHEMA = "neural-forge-hf-budget-ir-plan-v1"


class HFBudgetIRError(ValueError):
    """Base class for budget IR validation and compilation failures."""


class HFBudgetUnattainableError(HFBudgetIRError):
    """An exact or maximum budget has no representable plan."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)
        self.report = self.diagnostics


class HFBudgetCompilationError(HFBudgetIRError):
    """A model no longer matches a plan or compilation verification failed."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    unique_parameters: int
    mlp_group_count: int
    vocabulary_rank_slots: int

    def summary(self) -> dict[str, int]:
        return {
            "unique_parameters": self.unique_parameters,
            "mlp_group_count": self.mlp_group_count,
            "vocabulary_rank_slots": self.vocabulary_rank_slots,
        }


@dataclass(frozen=True, slots=True)
class _ModelCounts:
    layout: DecoderLayout
    embedding: nn.Embedding
    lm_head: nn.Linear
    mlp_modules: tuple[nn.Module, ...]
    mlp_layout: MLPModuleLayout
    logical_dense_parameters: int
    source_unique_parameters: int
    fixed_unique_parameters: int
    mlp_parameters_per_group: int
    vocabulary_parameters_per_rank: int
    vocabulary_bias_parameters: int
    vocabulary_tied: bool
    maximum_vocabulary_rank: int


@dataclass(frozen=True, slots=True)
class HFBudgetPlan:
    """Auditable storage plan selected from a finite deterministic search."""

    family: str
    model_class: str
    budget_mode: str
    requested_unique_parameter_budget: int
    logical_dense_parameters: int
    source_unique_parameters: int
    exact_unique_parameters: int
    execution_depth: int
    vocabulary_tied: bool
    embedding_rank: int
    lm_head_rank: int
    vocabulary_unique_parameters: int
    mlp_unique_parameters: int
    fixed_unique_parameters: int
    mlp_groups: tuple[tuple[int, ...], ...]
    shared_groups: tuple[tuple[int, ...], ...]
    vocabulary_plan: VocabularyFactorizationPlan
    diagnostics: Mapping[str, Any]

    @property
    def unique_parameter_count(self) -> int:
        return self.exact_unique_parameters

    @property
    def logical_dense_parameter_count(self) -> int:
        return self.logical_dense_parameters

    @property
    def factor_ranks(self) -> dict[str, int]:
        return {
            "embedding": self.embedding_rank,
            "lm_head": self.lm_head_rank,
        }

    @property
    def budget_headroom(self) -> int:
        return self.requested_unique_parameter_budget - self.exact_unique_parameters

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return {
            "schema": PLAN_SCHEMA,
            "family": self.family,
            "model_class": self.model_class,
            "budget_mode": self.budget_mode,
            "requested_unique_parameter_budget": (
                self.requested_unique_parameter_budget
            ),
            "logical_dense_parameters": self.logical_dense_parameters,
            "logical_dense_parameter_count": self.logical_dense_parameters,
            "source_unique_parameters": self.source_unique_parameters,
            "exact_unique_parameters": self.exact_unique_parameters,
            "unique_parameter_count": self.exact_unique_parameters,
            "budget_headroom": self.budget_headroom,
            "execution_depth": self.execution_depth,
            "all_logical_layers_executed": True,
            "vocabulary_tied": self.vocabulary_tied,
            "factor_ranks": self.factor_ranks,
            "vocabulary_unique_parameters": self.vocabulary_unique_parameters,
            "mlp_unique_parameters": self.mlp_unique_parameters,
            "fixed_unique_parameters": self.fixed_unique_parameters,
            "mlp_groups": [list(group) for group in self.mlp_groups],
            "shared_groups": [list(group) for group in self.shared_groups],
            "vocabulary_plan": self.vocabulary_plan.as_dict(),
            "diagnostics": dict(self.diagnostics),
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


class CompiledHFBudgetIR:
    """A compiled model plus the information needed for safe materialization."""

    def __init__(
        self,
        *,
        model: nn.Module,
        plan: HFBudgetPlan,
        vocabulary_pair: FactorizedVocabularyPair,
        embedding_path: str,
        lm_head_path: str,
        mlp_modules: Sequence[nn.Module],
        mlp_layout: MLPModuleLayout,
    ) -> None:
        self.model = model
        self.plan = plan
        self.vocabulary_pair = vocabulary_pair
        self.embedding_path = embedding_path
        self.lm_head_path = lm_head_path
        self.mlp_modules = tuple(mlp_modules)
        self.mlp_layout = mlp_layout
        self.materialized = False
        self.report = {
            **plan.as_dict(),
            "operation": "compile_hf_budget_ir",
            "actual_unique_parameters": plan.exact_unique_parameters,
            "planned_equals_actual": True,
        }

    @property
    def exact_unique_parameters(self) -> int:
        return int(
            exact_parameter_report(self.model)["unique_parameter_count"]
        )

    def materialize(self) -> nn.Module:
        return materialize_hf_budget_ir(self)


def _positive_budget(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFBudgetIRError(f"{name} must be a positive integer.")
    return value


def _resolve_budget(
    *,
    exact_unique_parameters: int | None,
    max_unique_parameters: int | None,
    exact_budget: int | None,
    max_budget: int | None,
) -> tuple[str, int]:
    supplied = [
        ("exact", exact_unique_parameters, "exact_unique_parameters"),
        ("max", max_unique_parameters, "max_unique_parameters"),
        ("exact", exact_budget, "exact_budget"),
        ("max", max_budget, "max_budget"),
    ]
    present = [(mode, value, name) for mode, value, name in supplied if value is not None]
    if len(present) != 1:
        raise HFBudgetIRError(
            "Supply exactly one exact or maximum unique-parameter budget."
        )
    mode, value, name = present[0]
    return mode, _positive_budget(value, name)


def _mlp_layout(layout: DecoderLayout) -> MLPModuleLayout:
    projection_bias = layout.blocks[0].mlp.projection_bias
    if layout.family == "gpt2":
        # GPT-2 Conv1D weights use [input, output].  Reversing only the
        # bookkeeping roles gives the sharing primitive its canonical shapes.
        return MLPModuleLayout(
            input_weight="c_proj.weight",
            down_weight="c_fc.weight",
            input_bias="c_fc.bias" if projection_bias else None,
            down_bias="c_proj.bias" if projection_bias else None,
        )
    if layout.family == "llama":
        return MLPModuleLayout.gated(
            "gate_proj",
            "up_proj",
            "down_proj",
            gate_bias=projection_bias,
            up_bias=projection_bias,
            down_bias=projection_bias,
        )
    raise HFBudgetIRError(f"Unsupported decoder family {layout.family!r}.")


def _inspect_dense_model(model: Any) -> _ModelCounts:
    if not isinstance(model, nn.Module):
        raise HFBudgetIRError("model must be a torch.nn.Module.")
    layout = build_decoder_layout(model)
    embedding = resolve_module(model, layout.embeddings.token_path, family=layout.family)
    lm_head = resolve_module(model, layout.lm_head_path, family=layout.family)
    if not isinstance(embedding, nn.Embedding) or not isinstance(lm_head, nn.Linear):
        raise HFBudgetIRError("Vocabulary operators must be dense before planning.")

    modules = tuple(
        resolve_module(model, block.mlp.module_path, family=layout.family)
        for block in layout.blocks
    )
    sharing_layout = _mlp_layout(layout)
    mlp_report = parameter_sharing_report(modules, sharing_layout)
    if not mlp_report.fully_expanded:
        raise HFBudgetIRError(
            "Planning requires independent dense MLP parameters in every layer."
        )
    if mlp_report.dense_equivalent_parameters % layout.layer_count:
        raise HFBudgetIRError("MLP parameter count is not uniform across layers.")

    total = exact_parameter_report(model)
    vocabulary = exact_parameter_report(embedding, lm_head)
    fixed_unique = (
        int(total["unique_parameter_count"])
        - int(vocabulary["unique_parameter_count"])
        - mlp_report.unique_parameters
    )
    if fixed_unique < 0:
        raise HFBudgetIRError("Model parameter categories overlap unexpectedly.")
    vocab_size = int(embedding.num_embeddings)
    hidden_size = int(embedding.embedding_dim)
    head_bias = 0 if lm_head.bias is None else int(lm_head.bias.numel())
    return _ModelCounts(
        layout=layout,
        embedding=embedding,
        lm_head=lm_head,
        mlp_modules=modules,
        mlp_layout=sharing_layout,
        logical_dense_parameters=int(total["logical_parameter_count"]),
        source_unique_parameters=int(total["unique_parameter_count"]),
        fixed_unique_parameters=fixed_unique,
        mlp_parameters_per_group=(
            mlp_report.dense_equivalent_parameters // layout.layer_count
        ),
        vocabulary_parameters_per_rank=vocab_size + hidden_size,
        vocabulary_bias_parameters=head_bias,
        vocabulary_tied=layout.tie_word_embeddings,
        maximum_vocabulary_rank=min(vocab_size, hidden_size),
    )


def _groups(depth: int, group_count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(index * depth // group_count, (index + 1) * depth // group_count))
        for index in range(group_count)
    )


def _candidate_key(candidate: _Candidate) -> tuple[int, int]:
    return candidate.mlp_group_count, candidate.vocabulary_rank_slots


def _search_candidates(
    counts: _ModelCounts,
    *,
    mode: str,
    budget: int,
) -> tuple[_Candidate | None, dict[str, Any]]:
    minimum_slots = 1 if counts.vocabulary_tied else 2
    maximum_slots = (
        counts.maximum_vocabulary_rank
        if counts.vocabulary_tied
        else counts.maximum_vocabulary_rank * 2
    )
    selected: _Candidate | None = None
    nearest_below: _Candidate | None = None
    nearest_above: _Candidate | None = None
    representable_counts: set[int] = set()
    candidate_count = 0

    for group_count in range(1, counts.layout.layer_count + 1):
        for rank_slots in range(minimum_slots, maximum_slots + 1):
            unique = (
                counts.fixed_unique_parameters
                + group_count * counts.mlp_parameters_per_group
                + counts.vocabulary_bias_parameters
                + rank_slots * counts.vocabulary_parameters_per_rank
            )
            candidate = _Candidate(unique, group_count, rank_slots)
            candidate_count += 1
            representable_counts.add(unique)
            if unique <= budget and (
                nearest_below is None
                or unique > nearest_below.unique_parameters
                or (
                    unique == nearest_below.unique_parameters
                    and _candidate_key(candidate) > _candidate_key(nearest_below)
                )
            ):
                nearest_below = candidate
            if unique >= budget and (
                nearest_above is None
                or unique < nearest_above.unique_parameters
                or (
                    unique == nearest_above.unique_parameters
                    and _candidate_key(candidate) > _candidate_key(nearest_above)
                )
            ):
                nearest_above = candidate
            eligible = unique == budget if mode == "exact" else unique <= budget
            if not eligible:
                continue
            if selected is None:
                selected = candidate
            elif mode == "max" and unique > selected.unique_parameters:
                selected = candidate
            elif unique == selected.unique_parameters and _candidate_key(
                candidate
            ) > _candidate_key(selected):
                selected = candidate

    sorted_counts = sorted(representable_counts)
    diagnostics: dict[str, Any] = {
        "deterministic": True,
        "search_order": "ascending-group-count-then-rank-slots",
        "tie_break": "more-independent-mlp-groups-then-more-rank-slots",
        "candidate_count": candidate_count,
        "representable_unique_parameter_count": len(sorted_counts),
        "minimum_representable_unique_parameters": sorted_counts[0],
        "maximum_representable_unique_parameters": sorted_counts[-1],
        "requested_unique_parameter_budget": budget,
        "exact_budget_representable": budget in representable_counts,
        "nearest_representable_below": (
            None if nearest_below is None else nearest_below.summary()
        ),
        "nearest_representable_above": (
            None if nearest_above is None else nearest_above.summary()
        ),
    }
    if len(sorted_counts) <= 1024:
        diagnostics["representable_unique_parameters"] = sorted_counts
        diagnostics["representable_unique_parameters_truncated"] = False
    else:
        diagnostics["representable_unique_parameters"] = []
        diagnostics["representable_unique_parameters_truncated"] = True
    return selected, diagnostics


def plan_hf_budget_ir(
    model: nn.Module,
    *,
    exact_unique_parameters: int | None = None,
    max_unique_parameters: int | None = None,
    exact_budget: int | None = None,
    max_budget: int | None = None,
) -> HFBudgetPlan:
    """Search deterministic vocabulary-rank and MLP-sharing plans."""

    mode, budget = _resolve_budget(
        exact_unique_parameters=exact_unique_parameters,
        max_unique_parameters=max_unique_parameters,
        exact_budget=exact_budget,
        max_budget=max_budget,
    )
    counts = _inspect_dense_model(model)
    selected, diagnostics = _search_candidates(counts, mode=mode, budget=budget)
    if selected is None:
        reason = (
            "exact_budget_not_representable"
            if mode == "exact"
            else "no_plan_at_or_below_max_budget"
        )
        failure = {**diagnostics, "reason": reason, "budget_mode": mode}
        raise HFBudgetUnattainableError(
            "The requested unique-parameter budget is not representable.",
            failure,
        )

    vocab_budget = (
        counts.vocabulary_bias_parameters
        + selected.vocabulary_rank_slots
        * counts.vocabulary_parameters_per_rank
    )
    if counts.vocabulary_tied:
        vocabulary_plan = plan_vocabulary_factorization(
            counts.embedding,
            counts.lm_head,
            rank=selected.vocabulary_rank_slots,
        )
    else:
        vocabulary_plan = plan_vocabulary_factorization(
            counts.embedding,
            counts.lm_head,
            parameter_budget=vocab_budget,
        )
    if vocabulary_plan.unique_parameter_count != vocab_budget:
        raise HFBudgetCompilationError(
            "Vocabulary primitive did not realize the searched parameter count."
        )

    groups = _groups(counts.layout.layer_count, selected.mlp_group_count)
    mlp_unique = selected.mlp_group_count * counts.mlp_parameters_per_group
    diagnostics = {
        **diagnostics,
        "selected": selected.summary(),
        "selected_at_or_below_budget": selected.unique_parameters <= budget,
        "selected_exactly_matches_budget": selected.unique_parameters == budget,
    }
    return HFBudgetPlan(
        family=counts.layout.family,
        model_class=counts.layout.model_class,
        budget_mode=mode,
        requested_unique_parameter_budget=budget,
        logical_dense_parameters=counts.logical_dense_parameters,
        source_unique_parameters=counts.source_unique_parameters,
        exact_unique_parameters=selected.unique_parameters,
        execution_depth=counts.layout.layer_count,
        vocabulary_tied=counts.vocabulary_tied,
        embedding_rank=vocabulary_plan.embedding.rank,
        lm_head_rank=vocabulary_plan.lm_head.rank,
        vocabulary_unique_parameters=vocab_budget,
        mlp_unique_parameters=mlp_unique,
        fixed_unique_parameters=counts.fixed_unique_parameters,
        mlp_groups=groups,
        shared_groups=tuple(group for group in groups if len(group) > 1),
        vocabulary_plan=vocabulary_plan,
        diagnostics=diagnostics,
    )


def _parameter_roles(layout: MLPModuleLayout) -> tuple[str, ...]:
    names = (
        "input_weight",
        "gate_weight",
        "up_weight",
        "down_weight",
        "input_bias",
        "gate_bias",
        "up_bias",
        "down_bias",
    )
    return tuple(name for name in names if getattr(layout, name) is not None)


def _parameter_slot(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    owner_path, separator, name = path.rpartition(".")
    owner = module.get_submodule(owner_path) if separator else module
    if name not in owner._parameters or owner._parameters[name] is None:
        raise HFBudgetCompilationError(f"{path!r} is not a registered parameter.")
    return owner, name


def _capture_mlp_parameters(
    modules: Sequence[nn.Module], layout: MLPModuleLayout
) -> tuple[tuple[nn.Module, str, nn.Parameter], ...]:
    captured = []
    for module in modules:
        for role in _parameter_roles(layout):
            owner, name = _parameter_slot(module, getattr(layout, role))
            captured.append((owner, name, owner._parameters[name]))
    return tuple(captured)


def _restore_parameters(
    captured: Sequence[tuple[nn.Module, str, nn.Parameter]],
) -> None:
    for owner, name, parameter in captured:
        owner._parameters[name] = parameter


def _set_module(model: nn.Module, path: str, module: nn.Module) -> None:
    owner_path, separator, name = path.rpartition(".")
    owner = resolve_module(model, owner_path) if separator else model
    if not isinstance(owner, nn.Module):
        raise HFBudgetCompilationError(f"Owner of {path!r} is not a module.")
    setattr(owner, name, module)


def _validate_plan(counts: _ModelCounts, plan: HFBudgetPlan) -> None:
    mismatches = []
    checks = {
        "family": (counts.layout.family, plan.family),
        "model_class": (counts.layout.model_class, plan.model_class),
        "execution_depth": (counts.layout.layer_count, plan.execution_depth),
        "logical_dense_parameters": (
            counts.logical_dense_parameters,
            plan.logical_dense_parameters,
        ),
        "source_unique_parameters": (
            counts.source_unique_parameters,
            plan.source_unique_parameters,
        ),
        "vocabulary_tied": (counts.vocabulary_tied, plan.vocabulary_tied),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            mismatches.append(f"{name}: expected {expected!r}, got {actual!r}")
    flattened = tuple(index for group in plan.mlp_groups for index in group)
    if flattened != tuple(range(counts.layout.layer_count)):
        mismatches.append("mlp_groups do not partition every decoder layer once")
    if mismatches:
        raise HFBudgetCompilationError(
            "Model does not match the budget plan: " + "; ".join(mismatches)
        )


def compile_hf_budget_ir(
    model: nn.Module,
    plan: HFBudgetPlan | None = None,
    *,
    exact_unique_parameters: int | None = None,
    max_unique_parameters: int | None = None,
    exact_budget: int | None = None,
    max_budget: int | None = None,
) -> CompiledHFBudgetIR:
    """Compile a plan in place and verify exact physical parameter storage."""

    if plan is None:
        plan = plan_hf_budget_ir(
            model,
            exact_unique_parameters=exact_unique_parameters,
            max_unique_parameters=max_unique_parameters,
            exact_budget=exact_budget,
            max_budget=max_budget,
        )
    elif any(
        value is not None
        for value in (
            exact_unique_parameters,
            max_unique_parameters,
            exact_budget,
            max_budget,
        )
    ):
        raise HFBudgetIRError("Do not supply a budget together with an explicit plan.")
    if not isinstance(plan, HFBudgetPlan):
        raise HFBudgetIRError("plan must be an HFBudgetPlan.")

    counts = _inspect_dense_model(model)
    _validate_plan(counts, plan)
    if counts.vocabulary_tied:
        pair = factorize_embedding_and_lm_head(
            counts.embedding,
            counts.lm_head,
            rank=plan.embedding_rank,
        )
    else:
        pair = factorize_embedding_and_lm_head(
            counts.embedding,
            counts.lm_head,
            parameter_budget=plan.vocabulary_unique_parameters,
        )
    if (
        pair.plan.embedding.rank != plan.embedding_rank
        or pair.plan.lm_head.rank != plan.lm_head_rank
        or pair.unique_parameter_count != plan.vocabulary_unique_parameters
    ):
        raise HFBudgetCompilationError(
            "Compiled vocabulary factors differ from the selected plan."
        )

    captured = _capture_mlp_parameters(counts.mlp_modules, counts.mlp_layout)
    original_embedding = counts.embedding
    original_head = counts.lm_head
    vocabulary_replaced = False
    try:
        for group in plan.shared_groups:
            modules = tuple(counts.mlp_modules[index] for index in group)
            bundles = tuple(
                extract_mlp_bundle(module, counts.mlp_layout) for module in modules
            )
            merged = merge_aligned_mlp_bundles(bundles)
            tie_mlp_modules(modules, merged, counts.mlp_layout)
        vocabulary_replaced = True
        _set_module(model, counts.layout.embeddings.token_path, pair.embedding)
        _set_module(model, counts.layout.lm_head_path, pair.lm_head)
        actual = int(exact_parameter_report(model)["unique_parameter_count"])
        if actual != plan.exact_unique_parameters:
            raise HFBudgetCompilationError(
                f"Planned {plan.exact_unique_parameters} unique parameters, "
                f"but compilation produced {actual}."
            )
        if len(counts.mlp_modules) != plan.execution_depth:
            raise HFBudgetCompilationError("Compilation changed decoder depth.")
    except Exception:
        _restore_parameters(captured)
        if vocabulary_replaced:
            _set_module(model, counts.layout.embeddings.token_path, original_embedding)
            _set_module(model, counts.layout.lm_head_path, original_head)
        raise

    return CompiledHFBudgetIR(
        model=model,
        plan=plan,
        vocabulary_pair=pair,
        embedding_path=counts.layout.embeddings.token_path,
        lm_head_path=counts.layout.lm_head_path,
        mlp_modules=counts.mlp_modules,
        mlp_layout=counts.mlp_layout,
    )


def materialize_hf_budget_ir(compiled: CompiledHFBudgetIR) -> nn.Module:
    """Restore dense vocabulary modules and independent MLP parameters in place."""

    if not isinstance(compiled, CompiledHFBudgetIR):
        raise HFBudgetIRError("compiled must be a CompiledHFBudgetIR.")
    if compiled.materialized:
        raise HFBudgetIRError("This compiled budget IR is already materialized.")

    captured = _capture_mlp_parameters(compiled.mlp_modules, compiled.mlp_layout)
    old_embedding = resolve_module(compiled.model, compiled.embedding_path)
    old_head = resolve_module(compiled.model, compiled.lm_head_path)
    vocabulary_replaced = False
    try:
        for group in compiled.plan.shared_groups:
            modules = tuple(compiled.mlp_modules[index] for index in group)
            expand_mlp_modules(modules, compiled.mlp_layout)
        dense_embedding, dense_head = compiled.vocabulary_pair.materialize()
        vocabulary_replaced = True
        _set_module(compiled.model, compiled.embedding_path, dense_embedding)
        _set_module(compiled.model, compiled.lm_head_path, dense_head)
        report = exact_parameter_report(compiled.model)
        actual = int(report["unique_parameter_count"])
        if actual != compiled.plan.source_unique_parameters:
            raise HFBudgetCompilationError(
                "Dense materialization did not restore the source unique count: "
                f"expected {compiled.plan.source_unique_parameters}, got {actual}."
            )
        mlp_report = parameter_sharing_report(
            compiled.mlp_modules, compiled.mlp_layout
        )
        if not mlp_report.fully_expanded:
            raise HFBudgetCompilationError(
                "Dense materialization left shared MLP parameters."
            )
    except Exception:
        _restore_parameters(captured)
        if vocabulary_replaced:
            _set_module(compiled.model, compiled.embedding_path, old_embedding)
            _set_module(compiled.model, compiled.lm_head_path, old_head)
        raise

    compiled.materialized = True
    compiled.report["materialized_unique_parameters"] = actual
    compiled.report["materialized_dense_mlp_parameters_independent"] = True
    compiled.report["materialized_vocabulary_dense"] = True
    return compiled.model


plan_hf_budget = plan_hf_budget_ir
compile_hf_budget = compile_hf_budget_ir
materialize_hf_budget = materialize_hf_budget_ir
BudgetPlan = HFBudgetPlan
BudgetCompilation = CompiledHFBudgetIR
BudgetUnattainableError = HFBudgetUnattainableError


__all__ = [
    "BudgetCompilation",
    "BudgetPlan",
    "BudgetUnattainableError",
    "CompiledHFBudgetIR",
    "HFBudgetCompilationError",
    "HFBudgetIRError",
    "HFBudgetPlan",
    "HFBudgetUnattainableError",
    "PLAN_SCHEMA",
    "compile_hf_budget",
    "compile_hf_budget_ir",
    "materialize_hf_budget",
    "materialize_hf_budget_ir",
    "plan_hf_budget",
    "plan_hf_budget_ir",
]
