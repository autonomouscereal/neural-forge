from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch

from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    ModelCircuitGraph,
    ParameterSpec,
    canonical_sha256,
)
from neural_forge.dynamic_closed_loop_fold import CompileStage
from neural_forge.dynamic_fold_budget import (
    SEMANTIC_ATTENTION_QUERY_KEY,
    SEMANTIC_ATTENTION_WRITER,
)
from neural_forge.dynamic_semantic_circuit_stage import (
    DynamicSemanticCircuitStageError,
    SemanticCircuitCompilationResult,
    SemanticCircuitSelector,
    SemanticCircuitStageContract,
    SemanticCircuitStageRegistry,
    SemanticCircuitStageReport,
    SemanticCircuitWriteHandoff,
    default_semantic_parameter_classifier,
)


SHAPE = (2, 2)
DTYPE = "float32"


@dataclass(frozen=True)
class _GraphRow:
    node_id: str
    kind: CircuitKind | str
    path: str
    role: str
    storage_group: str
    shape: tuple[int, ...] = SHAPE
    dtype: str = DTYPE


def _graph(rows: tuple[_GraphRow, ...], *, name: str = "synthetic") -> ModelCircuitGraph:
    nodes = tuple(
        CircuitNode(
            node_id=row.node_id,
            kind=row.kind,
            order=index,
            path=f"modules.{index}",
            layer_index=index,
            inputs=(),
            outputs=(),
            parameters=(
                ParameterSpec(
                    path=row.path,
                    role=row.role,
                    shape=row.shape,
                    dtype=row.dtype,
                    storage_group=row.storage_group,
                    trainable=False,
                ),
            ),
        )
        for index, row in enumerate(rows)
    )
    logical = sum(node.logical_parameter_count for node in nodes)
    unique = sum(
        next(
            parameter.numel
            for node in nodes
            for parameter in node.parameters
            if parameter.storage_group == group
        )
        for group in {row.storage_group for row in rows}
    )
    return ModelCircuitGraph(
        model_name=name,
        family="synthetic",
        model_class="SyntheticChild",
        nodes=nodes,
        edges=(),
        interfaces=(),
        logical_parameter_count=logical,
        unique_parameter_count=unique,
        unique_parameter_bytes=unique * 4,
        config_fingerprint=canonical_sha256(
            {"name": name, "paths": [row.path for row in rows]}
        ),
    )


PRODUCTION_ROWS = (
    _GraphRow(
        "token_embedding",
        CircuitKind.TOKEN_EMBEDDING,
        "model.embed.weight",
        "embedding.weight",
        "storage.embed",
    ),
    _GraphRow(
        "output_head",
        CircuitKind.OUTPUT_HEAD,
        "model.head.weight",
        "output.weight",
        "storage.head",
    ),
    _GraphRow(
        "attention",
        CircuitKind.ATTENTION,
        "model.attention.q.weight",
        "q_proj.weight",
        "storage.attention",
    ),
    _GraphRow(
        "mlp",
        CircuitKind.GATED_MLP,
        "model.mlp.down.weight",
        "down_proj.weight",
        "storage.mlp",
    ),
)


class _TensorChild:
    def __init__(self, values: dict[str, torch.nn.Parameter]) -> None:
        self.values = dict(values)

    @classmethod
    def blank(cls, graph: ModelCircuitGraph) -> _TensorChild:
        return cls(
            {
                parameter.path: torch.nn.Parameter(
                    torch.zeros(parameter.shape, dtype=torch.float32),
                    requires_grad=False,
                )
                for node in graph.nodes
                for parameter in node.parameters
            }
        )

    def named_parameters(
        self,
        *,
        recurse: bool = True,
        remove_duplicate: bool = False,
    ) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        del recurse, remove_duplicate
        return tuple(sorted(self.values.items()))


def _selector(kind: CircuitKind | str, *, optional: bool = False) -> SemanticCircuitSelector:
    return SemanticCircuitSelector(circuit_kinds=(kind,), optional=optional)


def _production_contracts(
    *,
    include_handoff: bool = True,
    effectful: bool = True,
) -> tuple[SemanticCircuitStageContract, ...]:
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    head = _selector(CircuitKind.OUTPUT_HEAD)
    attention = _selector(CircuitKind.ATTENTION)
    mlp = _selector(CircuitKind.GATED_MLP)
    structural = SemanticCircuitStageContract(
        stage_id="structural",
        stage_role=CompileStage.PRODUCER,
        dependencies=(),
        required_selectors=(),
        mutation_selectors=(embed, head),
        receipt_partitions=(("token_input", embed),),
        requires_effectful_write=effectful,
    )
    body = SemanticCircuitStageContract(
        stage_id="body",
        stage_role=CompileStage.DOWNSTREAM,
        dependencies=("structural",),
        required_selectors=(embed,),
        mutation_selectors=(attention, mlp),
        receipt_partitions=(
            ("attention", attention),
            ("mlp", mlp),
        ),
        requires_effectful_write=effectful,
    )
    causal = SemanticCircuitStageContract(
        stage_id="causal",
        stage_role=CompileStage.OUTPUT_INTERFACE,
        dependencies=("body", "structural"),
        required_selectors=(attention, mlp),
        mutation_selectors=(head,),
        receipt_partitions=(("token_output", head),),
        write_handoffs=(
            (SemanticCircuitWriteHandoff("structural", (head,)),)
            if include_handoff
            else ()
        ),
        requires_effectful_write=effectful,
    )
    return structural, body, causal


def _resource(stage_id: str) -> str:
    return canonical_sha256({"kind": "synthetic-stage-resource", "stage": stage_id})


def _single_embed_contract() -> SemanticCircuitStageContract:
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    return SemanticCircuitStageContract(
        stage_id="structural",
        stage_role=CompileStage.PRODUCER,
        dependencies=(),
        required_selectors=(),
        mutation_selectors=(embed,),
        receipt_partitions=(("embed", embed),),
    )


class _Compiler:
    def __init__(
        self,
        stage_id: str,
        *,
        mutations: tuple[str, ...] = (),
        delta: float = 1.0,
        mutate_resource: bool = False,
        replace_path: str | None = None,
    ) -> None:
        self._resource_identity = _resource(stage_id)
        self._mutations = mutations
        self._delta = delta
        self._mutate_resource = mutate_resource
        self._replace_path = replace_path
        self._calls: list[tuple[int, int, str, tuple[str, ...]]] = []

    @property
    def semantic_stage_resource_identity(self) -> str:
        return self._resource_identity

    @property
    def calls(self) -> tuple[tuple[int, int, str, tuple[str, ...]], ...]:
        return tuple(self._calls)

    def compile_stage(
        self,
        request: Any,
        shared_workspace: Any,
        prior_stage_results: Any,
    ) -> SemanticCircuitStageReport:
        assert isinstance(prior_stage_results, MappingProxyType)
        self._calls.append(
            (
                id(request.child_model),
                id(shared_workspace),
                request.stage_id,
                tuple(prior_stage_results),
            )
        )
        with torch.no_grad():
            for path in self._mutations:
                request.child_model.values[path].add_(self._delta)
        if self._replace_path is not None:
            incumbent = request.child_model.values[self._replace_path]
            request.child_model.values[self._replace_path] = torch.nn.Parameter(
                incumbent.detach().clone(),
                requires_grad=False,
            )
        if self._mutate_resource:
            self._resource_identity = canonical_sha256({"mutated": request.stage_id})
        return SemanticCircuitStageReport(
            partition_receipts=tuple(
                (
                    partition.partition_id,
                    {"stage_id": request.stage_id, "partition": partition.partition_id},
                )
                for partition in request.final_partitions
            ),
            operation_audit={
                "training_steps": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "domains": {"training": 0, "parent_execution": 0},
            },
            stage_receipt={"stage_id": request.stage_id},
        )


class _ResourceGetterMutationCompiler(_Compiler):
    def __init__(
        self,
        stage_id: str,
        *,
        child: _TensorChild,
        mutate_on_read: int,
    ) -> None:
        super().__init__(stage_id)
        self._child = child
        self._mutate_on_read = mutate_on_read
        self._resource_reads = 0

    @property
    def semantic_stage_resource_identity(self) -> str:
        self._resource_reads += 1
        if self._resource_reads == self._mutate_on_read:
            with torch.no_grad():
                self._child.values["model.embed.weight"].add_(1.0)
        return self._resource_identity


class _CaptureMutationChild(_TensorChild):
    def __init__(
        self,
        values: dict[str, torch.nn.Parameter],
        *,
        mutate_on_capture: int,
        mutation_path: str,
    ) -> None:
        super().__init__(values)
        self._capture_count = 0
        self._mutate_on_capture = mutate_on_capture
        self._mutation_path = mutation_path

    def named_parameters(
        self,
        *,
        recurse: bool = True,
        remove_duplicate: bool = False,
    ) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        self._capture_count += 1
        if self._capture_count == self._mutate_on_capture:
            with torch.no_grad():
                self.values[self._mutation_path].add_(1.0)
        return super().named_parameters(
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )


class _ReportSubclass(SemanticCircuitStageReport):
    pass


class _ReportSubclassCompiler(_Compiler):
    def compile_stage(
        self,
        request: Any,
        shared_workspace: Any,
        prior_stage_results: Any,
    ) -> SemanticCircuitStageReport:
        report = super().compile_stage(
            request,
            shared_workspace,
            prior_stage_results,
        )
        return _ReportSubclass(
            partition_receipts=report.partition_receipts,
            operation_audit=report.operation_audit,
            stage_receipt=report.stage_receipt,
        )


class _MutatingPartitionReceipts:
    def __init__(
        self,
        child: _TensorChild,
        values: tuple[tuple[str, Any], ...],
    ) -> None:
        self._child = child
        self._values = values
        self._fired = False

    def __iter__(self) -> Any:
        if not self._fired:
            self._fired = True
            with torch.no_grad():
                self._child.values["model.embed.weight"].add_(1.0)
        return iter(self._values)


class _ReportMaterializationMutationCompiler(_Compiler):
    def compile_stage(
        self,
        request: Any,
        shared_workspace: Any,
        prior_stage_results: Any,
    ) -> SemanticCircuitStageReport:
        report = super().compile_stage(
            request,
            shared_workspace,
            prior_stage_results,
        )
        object.__setattr__(
            report,
            "partition_receipts",
            _MutatingPartitionReceipts(
                request.child_model,
                report.partition_receipts,
            ),
        )
        return report


class _StageAuthorizationMutationCompiler(_Compiler):
    def compile_stage(
        self,
        request: Any,
        shared_workspace: Any,
        prior_stage_results: Any,
    ) -> SemanticCircuitStageReport:
        object.__setattr__(
            request.stage,
            "write_node_ids",
            tuple(sorted((*request.stage.write_node_ids, "output_head"))),
        )
        object.__setattr__(
            request.stage,
            "write_parameter_paths",
            tuple(
                sorted(
                    (
                        *request.stage.write_parameter_paths,
                        "model.head.weight",
                    )
                )
            ),
        )
        object.__setattr__(
            request.stage,
            "write_storage_groups",
            tuple(
                sorted(
                    (*request.stage.write_storage_groups, "storage.head")
                )
            ),
        )
        object.__setattr__(
            request.stage,
            "stage_fingerprint",
            canonical_sha256(request.stage.as_dict()),
        )
        with torch.no_grad():
            request.child_model.values["model.head.weight"].add_(1.0)
        return super().compile_stage(
            request,
            shared_workspace,
            prior_stage_results,
        )


class _StageAliasDivergenceCompiler(_Compiler):
    def compile_stage(
        self,
        request: Any,
        shared_workspace: Any,
        prior_stage_results: Any,
    ) -> SemanticCircuitStageReport:
        retained_stage = request.stage
        replacement_stage = replace(retained_stage)
        stage_index = request.manifest.ordered_stage_ids.index(request.stage_id)
        replacement_stages = list(request.manifest.stages)
        replacement_stages[stage_index] = replacement_stage
        object.__setattr__(
            request.manifest,
            "stages",
            tuple(replacement_stages),
        )
        object.__setattr__(
            retained_stage,
            "write_parameter_paths",
            tuple(
                sorted(
                    (
                        *retained_stage.write_parameter_paths,
                        "model.head.weight",
                    )
                )
            ),
        )
        object.__setattr__(
            retained_stage,
            "write_storage_groups",
            tuple(
                sorted(
                    (*retained_stage.write_storage_groups, "storage.head")
                )
            ),
        )
        object.__setattr__(
            retained_stage,
            "stage_fingerprint",
            canonical_sha256(retained_stage.as_dict()),
        )
        with torch.no_grad():
            request.child_model.values["model.head.weight"].add_(1.0)
        return super().compile_stage(
            request,
            shared_workspace,
            prior_stage_results,
        )


class _ManifestGraphMutationCompiler(_Compiler):
    def __init__(self, stage_id: str, *, fault: str) -> None:
        super().__init__(stage_id)
        self._fault = fault

    def compile_stage(
        self,
        request: Any,
        shared_workspace: Any,
        prior_stage_results: Any,
    ) -> SemanticCircuitStageReport:
        if self._fault == "graph":
            object.__setattr__(
                request.child_graph,
                "model_name",
                "forged_graph",
            )
            object.__setattr__(
                request.manifest,
                "child_graph_fingerprint",
                request.child_graph.fingerprint,
            )
        else:
            object.__setattr__(
                request.manifest,
                "initial_state_fingerprint",
                canonical_sha256({"forged": "initial-state"}),
            )
        object.__setattr__(
            request.manifest,
            "manifest_fingerprint",
            canonical_sha256(request.manifest.as_dict()),
        )
        return super().compile_stage(
            request,
            shared_workspace,
            prior_stage_results,
        )


class _ActionPartitionReceipts:
    def __init__(
        self,
        action: Any,
        values: tuple[tuple[str, Any], ...],
    ) -> None:
        self._action = action
        self._values = values
        self._fired = False

    def __iter__(self) -> Any:
        if not self._fired:
            self._fired = True
            self._action()
        return iter(self._values)


class _PriorResultMaterializationMutationCompiler(_Compiler):
    def compile_stage(
        self,
        request: Any,
        shared_workspace: Any,
        prior_stage_results: Any,
    ) -> SemanticCircuitStageReport:
        prior_result = prior_stage_results["structural"]
        report = super().compile_stage(
            request,
            shared_workspace,
            prior_stage_results,
        )

        def mutate_prior_result() -> None:
            object.__setattr__(
                prior_result,
                "stage_receipt",
                MappingProxyType({"forged": True}),
            )
            object.__setattr__(
                prior_result,
                "result_fingerprint",
                canonical_sha256(prior_result.as_dict()),
            )

        object.__setattr__(
            report,
            "partition_receipts",
            _ActionPartitionReceipts(
                mutate_prior_result,
                report.partition_receipts,
            ),
        )
        return report


def _registry(
    contracts: tuple[SemanticCircuitStageContract, ...],
    compilers: tuple[_Compiler, ...],
    **kwargs: Any,
) -> SemanticCircuitStageRegistry:
    registry = SemanticCircuitStageRegistry(**kwargs)
    for contract, compiler in zip(contracts, compilers, strict=True):
        registry.register(
            contract,
            compiler,
            resource_identity=compiler.semantic_stage_resource_identity,
        )
    return registry


def _production_compilers() -> tuple[_Compiler, ...]:
    return (
        _Compiler(
            "structural",
            mutations=("model.embed.weight", "model.head.weight"),
        ),
        _Compiler(
            "body",
            mutations=("model.attention.q.weight", "model.mlp.down.weight"),
        ),
        _Compiler("causal", mutations=("model.head.weight",), delta=2.0),
    )


def test_valid_ordered_handoff_observes_real_lfm2_style_output_rewrite() -> None:
    graph = _graph(PRODUCTION_ROWS)
    child = _TensorChild.blank(graph)
    contracts = _production_contracts()
    compilers = _production_compilers()
    registry = _registry(contracts, compilers)
    workspace: dict[str, Any] = {}

    result = registry.compile(
        child_model=child,
        child_graph=graph,
        shared_workspace=workspace,
    )

    assert isinstance(result, SemanticCircuitCompilationResult)
    assert result.manifest.ordered_stage_ids == ("structural", "body", "causal")
    assert dict(
        (group, stage)
        for group, stage, _partition in result.manifest.final_storage_owners
    ) == {
        "storage.embed": "structural",
        "storage.head": "causal",
        "storage.attention": "body",
        "storage.mlp": "body",
    }
    assert result.stage_result("structural").observed_changed_storage_groups == (
        "storage.embed",
        "storage.head",
    )
    assert result.stage_result("causal").observed_changed_storage_groups == (
        "storage.head",
    )
    assert torch.equal(
        child.values["model.head.weight"],
        torch.full(SHAPE, 3.0),
    )
    assert all(
        call[0] == id(child) and call[1] == id(workspace)
        for compiler in compilers
        for call in compiler.calls
    )
    assert compilers[-1].calls[0][-1] == ("body", "structural")
    assert result.manifest.as_dict()["stages"][0]["contract"]["stage_role"] == (
        CompileStage.PRODUCER.value
    )


def test_overwrite_without_explicit_handoff_fails_before_any_callback() -> None:
    graph = _graph(PRODUCTION_ROWS)
    child = _TensorChild.blank(graph)
    compilers = _production_compilers()
    registry = _registry(
        _production_contracts(include_handoff=False),
        compilers,
    )

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="lack the explicit adjacent handoff",
    ):
        registry.compile(child_model=child, child_graph=graph, shared_workspace={})

    assert all(not compiler.calls for compiler in compilers)


def test_rogue_storage_write_is_detected_and_outer_caller_owns_rollback() -> None:
    graph = _graph(PRODUCTION_ROWS)
    child = _TensorChild.blank(graph)
    structural, body, causal = _production_contracts()
    compilers = (
        _Compiler(
            "structural",
            mutations=("model.embed.weight", "model.head.weight"),
        ),
        _Compiler("body", mutations=("model.head.weight",)),
        _Compiler("causal", mutations=("model.head.weight",)),
    )
    registry = _registry((structural, body, causal), compilers)

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="performed an unauthorized learned write.*storage.head",
    ):
        registry.compile(child_model=child, child_graph=graph, shared_workspace={})

    assert torch.equal(
        child.values["model.head.weight"],
        torch.full(SHAPE, 2.0),
    )
    assert not compilers[-1].calls
    for forbidden_surface in (
        "accept",
        "commit",
        "publish",
        "reject",
        "rollback",
        "skip",
    ):
        assert not hasattr(registry, forbidden_surface)


def test_noop_stage_is_valid_unless_contract_explicitly_requires_effect() -> None:
    graph = _graph(PRODUCTION_ROWS)
    child = _TensorChild.blank(graph)
    contracts = _production_contracts(effectful=False)
    compilers = tuple(_Compiler(contract.stage_id) for contract in contracts)
    result = _registry(contracts, compilers).compile(
        child_model=child,
        child_graph=graph,
        shared_workspace={},
    )
    assert all(not value.observed_writes for value in result.stage_results)
    assert all(
        value.partition_receipts or value.stage_id == "causal"
        for value in result.stage_results
    )

    effectful_contracts = _production_contracts(effectful=True)
    effectful_compilers = tuple(
        _Compiler(contract.stage_id) for contract in effectful_contracts
    )
    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="Effectful stage 'structural'.*unchanged",
    ):
        _registry(effectful_contracts, effectful_compilers).compile(
            child_model=_TensorChild.blank(graph),
            child_graph=graph,
            shared_workspace={},
        )


def _tied_graph() -> ModelCircuitGraph:
    return _graph(
        (
            _GraphRow(
                "token_embedding",
                CircuitKind.TOKEN_EMBEDDING,
                "model.embed.weight",
                "embedding.weight",
                "storage.shared_token",
            ),
            _GraphRow(
                "output_head",
                CircuitKind.OUTPUT_HEAD,
                "model.head.weight",
                "output.weight",
                "storage.shared_token",
            ),
        ),
        name="tied",
    )


def _tied_contracts() -> tuple[SemanticCircuitStageContract, ...]:
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    head = _selector(CircuitKind.OUTPUT_HEAD)
    return (
        SemanticCircuitStageContract(
            stage_id="structural",
            stage_role=CompileStage.PRODUCER,
            dependencies=(),
            required_selectors=(),
            mutation_selectors=(embed,),
            receipt_partitions=(),
            requires_effectful_write=True,
        ),
        SemanticCircuitStageContract(
            stage_id="causal",
            stage_role=CompileStage.OUTPUT_INTERFACE,
            dependencies=("structural",),
            required_selectors=(head,),
            mutation_selectors=(head,),
            receipt_partitions=(("shared_token", head),),
            write_handoffs=(
                SemanticCircuitWriteHandoff("structural", (head,)),
            ),
            requires_effectful_write=True,
        ),
    )


def test_tied_alias_selection_closes_over_every_path_and_observes_both() -> None:
    graph = _tied_graph()
    shared = torch.nn.Parameter(torch.zeros(SHAPE), requires_grad=False)
    child = _TensorChild(
        {"model.embed.weight": shared, "model.head.weight": shared}
    )
    contracts = _tied_contracts()
    compilers = (
        _Compiler("structural", mutations=("model.embed.weight",)),
        _Compiler("causal", mutations=("model.head.weight",)),
    )

    result = _registry(contracts, compilers).compile(
        child_model=child,
        child_graph=graph,
        shared_workspace={},
    )

    structural = result.manifest.stages[0]
    assert structural.write_parameter_paths == (
        "model.embed.weight",
        "model.head.weight",
    )
    assert result.stage_result(
        "structural"
    ).observed_changed_parameter_paths == (
        "model.embed.weight",
        "model.head.weight",
    )
    final_receipt = result.stage_result("causal").partition_receipts[0]
    assert final_receipt.storage_groups == ("storage.shared_token",)
    assert final_receipt.parameter_paths == (
        "model.embed.weight",
        "model.head.weight",
    )


@pytest.mark.parametrize("fault", ["shape", "dtype", "split_tie", "merged_tie"])
def test_live_shape_dtype_and_tie_topology_must_match_graph(fault: str) -> None:
    if fault in {"split_tie", "shape", "dtype"}:
        graph = _tied_graph()
        shared = torch.nn.Parameter(torch.zeros(SHAPE), requires_grad=False)
        values = {"model.embed.weight": shared, "model.head.weight": shared}
        if fault == "split_tie":
            values["model.head.weight"] = torch.nn.Parameter(
                torch.zeros(SHAPE),
                requires_grad=False,
            )
        elif fault == "shape":
            values["model.head.weight"] = torch.nn.Parameter(
                torch.zeros((2, 3)),
                requires_grad=False,
            )
        elif fault == "dtype":
            values["model.head.weight"] = torch.nn.Parameter(
                torch.zeros(SHAPE, dtype=torch.float64),
                requires_grad=False,
            )
        contracts = _tied_contracts()
    else:
        graph = _graph(PRODUCTION_ROWS[:2], name="untied")
        shared = torch.nn.Parameter(torch.zeros(SHAPE), requires_grad=False)
        values = {"model.embed.weight": shared, "model.head.weight": shared}
        embed = _selector(CircuitKind.TOKEN_EMBEDDING)
        head = _selector(CircuitKind.OUTPUT_HEAD)
        contracts = (
            SemanticCircuitStageContract(
                stage_id="interface",
                stage_role=CompileStage.OUTPUT_INTERFACE,
                dependencies=(),
                required_selectors=(),
                mutation_selectors=(embed, head),
                receipt_partitions=(("embed", embed), ("head", head)),
            ),
        )
    compilers = tuple(_Compiler(contract.stage_id) for contract in contracts)
    registry = _registry(contracts, compilers)

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match=("shape|dtype|tie/storage topology"),
    ):
        registry.preflight(_TensorChild(values), graph)


def test_parameter_object_replacement_is_rejected_without_inner_rollback() -> None:
    graph = _graph(PRODUCTION_ROWS[:2], name="object-replacement")
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    head = _selector(CircuitKind.OUTPUT_HEAD)
    contract = SemanticCircuitStageContract(
        stage_id="interface",
        stage_role=CompileStage.OUTPUT_INTERFACE,
        dependencies=(),
        required_selectors=(),
        mutation_selectors=(embed, head),
        receipt_partitions=(("embed", embed), ("head", head)),
    )
    child = _TensorChild.blank(graph)
    original = child.values["model.head.weight"]
    compiler = _Compiler("interface", replace_path="model.head.weight")

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="Parameter object identity changed",
    ):
        _registry((contract,), (compiler,)).compile(
            child_model=child,
            child_graph=graph,
            shared_workspace={},
        )

    assert child.values["model.head.weight"] is not original


def test_callback_resource_identity_is_reverified_immediately_after_call() -> None:
    graph = _graph(PRODUCTION_ROWS[:1], name="resource-toctou")
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    contract = SemanticCircuitStageContract(
        stage_id="structural",
        stage_role=CompileStage.PRODUCER,
        dependencies=(),
        required_selectors=(),
        mutation_selectors=(embed,),
        receipt_partitions=(("embed", embed),),
    )
    child = _TensorChild.blank(graph)
    compiler = _Compiler(
        "structural",
        mutations=("model.embed.weight",),
        mutate_resource=True,
    )

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="explicit compiler resource identity changed",
    ):
        _registry((contract,), (compiler,)).compile(
            child_model=child,
            child_graph=graph,
            shared_workspace={},
        )

    assert torch.equal(
        child.values["model.embed.weight"],
        torch.ones(SHAPE),
    )


@pytest.mark.parametrize(
    ("mutate_on_read", "phase", "expected_callback_count"),
    (
        (5, "before", 0),
        (7, "after", 1),
    ),
)
def test_resource_getter_writes_are_not_attributed_to_the_callback(
    mutate_on_read: int,
    phase: str,
    expected_callback_count: int,
) -> None:
    graph = _graph(PRODUCTION_ROWS[:1], name=f"resource-write-{phase}")
    child = _TensorChild.blank(graph)
    compiler = _ResourceGetterMutationCompiler(
        "structural",
        child=child,
        mutate_on_read=mutate_on_read,
    )

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match=rf"authentication mutated learned child state {phase}",
    ):
        _registry((_single_embed_contract(),), (compiler,)).compile(
            child_model=child,
            child_graph=graph,
            shared_workspace={},
        )

    assert len(compiler.calls) == expected_callback_count
    assert torch.equal(child.values["model.embed.weight"], torch.ones(SHAPE))


class _CustomClassifier:
    def __init__(self) -> None:
        self._resource = canonical_sha256({"classifier": "hyena-decay-v1"})

    @property
    def semantic_classifier_resource_identity(self) -> str:
        return self._resource

    def __call__(self, circuit_kind: str, parameter_role: str) -> tuple[str, ...]:
        if circuit_kind == "hyena_mixer" and parameter_role == "decay.kernel":
            return ("hyena_decay",)
        return default_semantic_parameter_classifier(circuit_kind, parameter_role)


class _ChildMutatingClassifier:
    def __init__(self, child: _TensorChild) -> None:
        self._child = child
        self._mutated = False
        self._resource = canonical_sha256({"classifier": "mutating-v1"})

    @property
    def semantic_classifier_resource_identity(self) -> str:
        return self._resource

    def __call__(self, circuit_kind: str, parameter_role: str) -> tuple[str, ...]:
        if not self._mutated:
            self._mutated = True
            with torch.no_grad():
                self._child.values["model.embed.weight"].add_(1.0)
        return default_semantic_parameter_classifier(circuit_kind, parameter_role)


def test_classifier_write_during_preflight_is_rejected_before_callbacks() -> None:
    graph = _graph(PRODUCTION_ROWS[:1], name="classifier-write")
    child = _TensorChild.blank(graph)
    classifier = _ChildMutatingClassifier(child)
    compiler = _Compiler("structural")
    registry = _registry(
        (_single_embed_contract(),),
        (compiler,),
        semantic_classifier=classifier,
        semantic_classifier_resource_identity=(
            classifier.semantic_classifier_resource_identity
        ),
    )

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="mutated learned child state during preflight",
    ):
        registry.compile(
            child_model=child,
            child_graph=graph,
            shared_workspace={},
        )

    assert not compiler.calls
    assert torch.equal(child.values["model.embed.weight"], torch.ones(SHAPE))


def test_authenticated_custom_kind_and_pluggable_semantic_compile() -> None:
    graph = _graph(
        (
            _GraphRow(
                "hyena",
                "hyena_mixer",
                "model.hyena.decay",
                "decay.kernel",
                "storage.hyena",
            ),
        ),
        name="custom-kind",
    )
    selector = SemanticCircuitSelector(
        circuit_kinds=("hyena_mixer",),
        parameter_semantics=("hyena_decay",),
    )
    optional_absent = SemanticCircuitSelector(
        circuit_kinds=(CircuitKind.MODALITY_INPUT,),
        optional=True,
    )
    contract = SemanticCircuitStageContract(
        stage_id="hyena",
        stage_role=CompileStage.CONSUMER,
        dependencies=(),
        required_selectors=(optional_absent,),
        mutation_selectors=(selector,),
        receipt_partitions=(("hyena", selector),),
        requires_effectful_write=True,
    )
    classifier = _CustomClassifier()
    compiler = _Compiler("hyena", mutations=("model.hyena.decay",))
    registry = _registry(
        (contract,),
        (compiler,),
        semantic_classifier=classifier,
        semantic_classifier_resource_identity=(
            classifier.semantic_classifier_resource_identity
        ),
        custom_circuit_kinds=("hyena_mixer",),
        custom_parameter_semantics=("hyena_decay",),
    )

    result = registry.compile(
        child_model=_TensorChild.blank(graph),
        child_graph=graph,
        shared_workspace={},
    )
    assert result.manifest.stages[0].required_parameter_paths == ()
    assert result.stage_result("hyena").observed_changed_storage_groups == (
        "storage.hyena",
    )


def test_unauthenticated_custom_selector_vocabulary_is_rejected() -> None:
    graph = _graph(PRODUCTION_ROWS[:1], name="unknown-selector")
    selector = SemanticCircuitSelector(circuit_kinds=("typo_mixer",))
    contract = SemanticCircuitStageContract(
        stage_id="bad",
        stage_role=CompileStage.PRODUCER,
        dependencies=(),
        required_selectors=(),
        mutation_selectors=(selector,),
        receipt_partitions=(("bad", selector),),
    )
    compiler = _Compiler("bad")
    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="unauthenticated vocabulary",
    ):
        _registry((contract,), (compiler,)).preflight(
            _TensorChild.blank(graph),
            graph,
        )


def test_fused_multi_role_tensor_matches_every_semantic_without_paths() -> None:
    assert set(
        default_semantic_parameter_classifier(
            CircuitKind.ATTENTION.value,
            "q_proj/v_proj",
        )
    ) == {SEMANTIC_ATTENTION_QUERY_KEY, SEMANTIC_ATTENTION_WRITER}
    graph = _graph(
        (
            _GraphRow(
                "fused_attention",
                CircuitKind.ATTENTION,
                "model.fused.weight",
                "q_proj/v_proj",
                "storage.fused",
            ),
        ),
        name="fused-role",
    )
    query_key = SemanticCircuitSelector(
        parameter_semantics=(SEMANTIC_ATTENTION_QUERY_KEY,)
    )
    writer = SemanticCircuitSelector(
        parameter_semantics=(SEMANTIC_ATTENTION_WRITER,)
    )
    attention = _selector(CircuitKind.ATTENTION)
    contract = SemanticCircuitStageContract(
        stage_id="fused",
        stage_role=CompileStage.CONSUMER,
        dependencies=(),
        required_selectors=(query_key, writer),
        mutation_selectors=(query_key, writer),
        receipt_partitions=(("fused", attention),),
        requires_effectful_write=True,
    )
    result = _registry(
        (contract,),
        (_Compiler("fused", mutations=("model.fused.weight",)),),
    ).compile(
        child_model=_TensorChild.blank(graph),
        child_graph=graph,
        shared_workspace={},
    )
    stage = result.manifest.stages[0]
    assert stage.required_parameter_paths == ("model.fused.weight",)
    assert stage.write_parameter_paths == ("model.fused.weight",)


def test_typed_results_are_deeply_immutable_and_receipts_are_exact() -> None:
    graph = _graph(PRODUCTION_ROWS)
    result = _registry(
        _production_contracts(),
        _production_compilers(),
    ).compile(
        child_model=_TensorChild.blank(graph),
        child_graph=graph,
        shared_workspace={},
    )
    structural = result.stage_result("structural")
    assert structural.partition_receipts[0].partition_id == "token_input"
    assert structural.result_fingerprint == canonical_sha256(structural.as_dict())
    assert result.result_fingerprint == canonical_sha256(result.as_dict())
    assert isinstance(structural.operation_audit, MappingProxyType)
    assert isinstance(structural.operation_audit["domains"], MappingProxyType)
    with pytest.raises(TypeError):
        structural.operation_audit["training_steps"] = 9  # type: ignore[index]
    with pytest.raises(AttributeError):
        structural.stage_id = "changed"  # type: ignore[misc]


class _MissingReceiptCompiler(_Compiler):
    def compile_stage(
        self,
        request: Any,
        shared_workspace: Any,
        prior_stage_results: Any,
    ) -> SemanticCircuitStageReport:
        del request, shared_workspace, prior_stage_results
        return SemanticCircuitStageReport(
            partition_receipts=(),
            operation_audit={},
        )


def test_compiler_cannot_omit_or_self_define_final_partition_ownership() -> None:
    graph = _graph(PRODUCTION_ROWS[:1], name="receipt-mismatch")
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    contract = SemanticCircuitStageContract(
        stage_id="structural",
        stage_role=CompileStage.PRODUCER,
        dependencies=(),
        required_selectors=(),
        mutation_selectors=(embed,),
        receipt_partitions=(("embed", embed),),
    )
    compiler = _MissingReceiptCompiler("structural")
    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="report partition IDs differ",
    ):
        _registry((contract,), (compiler,)).compile(
            child_model=_TensorChild.blank(graph),
            child_graph=graph,
            shared_workspace={},
        )


def test_manifest_and_callable_binding_are_registration_order_deterministic() -> None:
    graph = _graph(PRODUCTION_ROWS)
    contracts = _production_contracts(effectful=False)
    left_compilers = tuple(_Compiler(contract.stage_id) for contract in contracts)
    right_compilers = tuple(_Compiler(contract.stage_id) for contract in contracts)
    left = _registry(contracts, left_compilers)
    right = SemanticCircuitStageRegistry()
    for contract, compiler in reversed(tuple(zip(contracts, right_compilers, strict=True))):
        right.register(
            contract,
            compiler,
            resource_identity=compiler.semantic_stage_resource_identity,
        )
    left_manifest = left.preflight(_TensorChild.blank(graph), graph)
    right_manifest = right.preflight(_TensorChild.blank(graph), graph)
    assert left.fingerprint == right.fingerprint
    assert left_manifest.as_dict() == right_manifest.as_dict()
    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="registry is frozen",
    ):
        left.register(
            contracts[0],
            _Compiler("structural"),
            resource_identity=_resource("structural"),
        )


def test_semantic_resolution_is_path_rename_independent() -> None:
    original = _graph(PRODUCTION_ROWS[:1], name="original-path")
    renamed = _graph(
        (
            _GraphRow(
                "token_embedding",
                CircuitKind.TOKEN_EMBEDDING,
                "completely.renamed.table",
                "embedding.weight",
                "storage.embed",
            ),
        ),
        name="renamed-path",
    )
    selector = _selector(CircuitKind.TOKEN_EMBEDDING)
    contract = SemanticCircuitStageContract(
        stage_id="structural",
        stage_role=CompileStage.PRODUCER,
        dependencies=(),
        required_selectors=(),
        mutation_selectors=(selector,),
        receipt_partitions=(("embed", selector),),
    )
    original_manifest = _registry(
        (contract,),
        (_Compiler("structural"),),
    ).preflight(_TensorChild.blank(original), original)
    renamed_manifest = _registry(
        (contract,),
        (_Compiler("structural"),),
    ).preflight(_TensorChild.blank(renamed), renamed)
    assert original_manifest.stages[0].write_parameter_paths == (
        "model.embed.weight",
    )
    assert renamed_manifest.stages[0].write_parameter_paths == (
        "completely.renamed.table",
    )
    assert "model.embed.weight" not in str(contract.as_dict())


def test_dependency_cycle_and_missing_final_storage_owner_fail_preflight() -> None:
    graph = _graph(PRODUCTION_ROWS[:2], name="preflight-failures")
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    head = _selector(CircuitKind.OUTPUT_HEAD)
    cyclic = (
        SemanticCircuitStageContract(
            stage_id="embed",
            stage_role=CompileStage.PRODUCER,
            dependencies=("head",),
            required_selectors=(),
            mutation_selectors=(embed,),
            receipt_partitions=(("embed", embed),),
        ),
        SemanticCircuitStageContract(
            stage_id="head",
            stage_role=CompileStage.OUTPUT_INTERFACE,
            dependencies=("embed",),
            required_selectors=(),
            mutation_selectors=(head,),
            receipt_partitions=(("head", head),),
        ),
    )
    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="dependencies contain a cycle",
    ):
        _registry(
            cyclic,
            (_Compiler("embed"), _Compiler("head")),
        ).preflight(_TensorChild.blank(graph), graph)

    missing_owner = SemanticCircuitStageContract(
        stage_id="embed",
        stage_role=CompileStage.PRODUCER,
        dependencies=(),
        required_selectors=(),
        mutation_selectors=(embed,),
        receipt_partitions=(("embed", embed),),
    )
    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="do not exactly own every learned storage group.*storage.head",
    ):
        _registry(
            (missing_owner,),
            (_Compiler("embed"),),
        ).preflight(_TensorChild.blank(graph), graph)


def test_learned_state_continuity_is_sealed_between_adjacent_stages() -> None:
    graph = _graph(PRODUCTION_ROWS[:2], name="between-stage-continuity")
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    head = _selector(CircuitKind.OUTPUT_HEAD)
    contracts = (
        _single_embed_contract(),
        SemanticCircuitStageContract(
            stage_id="causal",
            stage_role=CompileStage.OUTPUT_INTERFACE,
            dependencies=("structural",),
            required_selectors=(embed,),
            mutation_selectors=(head,),
            receipt_partitions=(("head", head),),
        ),
    )
    compilers = (_Compiler("structural"), _Compiler("causal"))
    blank = _TensorChild.blank(graph)
    child = _CaptureMutationChild(
        blank.values,
        mutate_on_capture=8,
        mutation_path="model.head.weight",
    )

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="continuity changed between adjacent stages",
    ):
        _registry(contracts, compilers).compile(
            child_model=child,
            child_graph=graph,
            shared_workspace={},
        )

    assert len(compilers[0].calls) == 1
    assert not compilers[1].calls
    assert torch.equal(child.values["model.head.weight"], torch.ones(SHAPE))


def test_report_subclass_cannot_escape_the_authenticated_report_surface() -> None:
    graph = _graph(PRODUCTION_ROWS[:1], name="report-subclass")
    compiler = _ReportSubclassCompiler("structural")

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="must return an exact SemanticCircuitStageReport",
    ):
        _registry((_single_embed_contract(),), (compiler,)).compile(
            child_model=_TensorChild.blank(graph),
            child_graph=graph,
            shared_workspace={},
        )


def test_final_live_recapture_seals_report_result_materialization() -> None:
    graph = _graph(PRODUCTION_ROWS[:1], name="report-materialization-write")
    child = _TensorChild.blank(graph)
    compiler = _ReportMaterializationMutationCompiler("structural")

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="report/result materialization mutated learned child state",
    ):
        _registry((_single_embed_contract(),), (compiler,)).compile(
            child_model=child,
            child_graph=graph,
            shared_workspace={},
        )

    assert torch.equal(child.values["model.embed.weight"], torch.ones(SHAPE))


def test_callback_cannot_expand_its_resolved_stage_write_authorization() -> None:
    graph = _graph(PRODUCTION_ROWS[:2], name="stage-authorization-mutation")
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    head = _selector(CircuitKind.OUTPUT_HEAD)
    contracts = (
        _single_embed_contract(),
        SemanticCircuitStageContract(
            stage_id="causal",
            stage_role=CompileStage.OUTPUT_INTERFACE,
            dependencies=("structural",),
            required_selectors=(embed,),
            mutation_selectors=(head,),
            receipt_partitions=(("head", head),),
        ),
    )
    child = _TensorChild.blank(graph)
    compilers = (
        _StageAuthorizationMutationCompiler("structural"),
        _Compiler("causal"),
    )

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match="after callback 'structural': .*changed",
    ):
        _registry(contracts, compilers).compile(
            child_model=child,
            child_graph=graph,
            shared_workspace={},
        )

    assert torch.equal(child.values["model.head.weight"], torch.ones(SHAPE))
    assert not compilers[1].calls


def test_replaced_manifest_stage_cannot_hide_retained_stage_divergence() -> None:
    graph = _graph(PRODUCTION_ROWS[:2], name="stage-alias-divergence")
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    head = _selector(CircuitKind.OUTPUT_HEAD)
    contracts = (
        _single_embed_contract(),
        SemanticCircuitStageContract(
            stage_id="causal",
            stage_role=CompileStage.OUTPUT_INTERFACE,
            dependencies=("structural",),
            required_selectors=(embed,),
            mutation_selectors=(head,),
            receipt_partitions=(("head", head),),
        ),
    )
    child = _TensorChild.blank(graph)
    compilers = (
        _StageAliasDivergenceCompiler("structural"),
        _Compiler("causal"),
    )

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match=(
            "retained pre-callback local stage.*diverged from the "
            "authenticated manifest stage"
        ),
    ):
        _registry(contracts, compilers).compile(
            child_model=child,
            child_graph=graph,
            shared_workspace={},
        )

    assert torch.equal(child.values["model.head.weight"], torch.ones(SHAPE))
    assert not compilers[1].calls


@pytest.mark.parametrize("fault", ("manifest", "graph"))
def test_callback_cannot_reseal_a_mutated_manifest_or_graph(fault: str) -> None:
    graph = _graph(PRODUCTION_ROWS[:1], name=f"control-plane-{fault}")
    compiler = _ManifestGraphMutationCompiler("structural", fault=fault)
    expected = (
        "manifest body/fingerprint changed"
        if fault == "manifest"
        else "child graph or manifest graph binding changed"
    )

    with pytest.raises(DynamicSemanticCircuitStageError, match=expected):
        _registry((_single_embed_contract(),), (compiler,)).compile(
            child_model=_TensorChild.blank(graph),
            child_graph=graph,
            shared_workspace={},
        )


def test_report_materialization_cannot_reseal_a_mutated_prior_result() -> None:
    graph = _graph(PRODUCTION_ROWS[:2], name="prior-result-mutation")
    embed = _selector(CircuitKind.TOKEN_EMBEDDING)
    head = _selector(CircuitKind.OUTPUT_HEAD)
    contracts = (
        _single_embed_contract(),
        SemanticCircuitStageContract(
            stage_id="causal",
            stage_role=CompileStage.OUTPUT_INTERFACE,
            dependencies=("structural",),
            required_selectors=(embed,),
            mutation_selectors=(head,),
            receipt_partitions=(("head", head),),
        ),
    )
    compilers = (
        _Compiler("structural"),
        _PriorResultMaterializationMutationCompiler("causal"),
    )

    with pytest.raises(
        DynamicSemanticCircuitStageError,
        match=(
            "after report materialization 'causal': prior stage result "
            "'structural' body/fingerprint changed"
        ),
    ):
        _registry(contracts, compilers).compile(
            child_model=_TensorChild.blank(graph),
            child_graph=graph,
            shared_workspace={},
        )


def test_handoff_rows_are_deterministic_across_python_hash_seeds() -> None:
    script = "\n".join(
        (
            "import json",
            (
                "from neural_forge.dynamic_semantic_circuit_stage import "
                "SemanticCircuitResolvedHandoff"
            ),
            "paths = {'model.z.weight', 'model.a.weight', 'model.m.weight'}",
            "groups = {'storage.z', 'storage.a', 'storage.m'}",
            "row = SemanticCircuitResolvedHandoff(",
            "    from_stage_id='structural',",
            "    parameter_paths=tuple(paths),",
            "    storage_groups=tuple(groups),",
            ")",
            "print(json.dumps(row.as_dict(), sort_keys=True))",
        )
    )
    outputs: list[str] = []
    for seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        outputs.append(completed.stdout.strip())

    assert outputs[0] == outputs[1]
    assert outputs[0] == (
        '{"from_stage_id": "structural", '
        '"parameter_paths": ["model.a.weight", "model.m.weight", '
        '"model.z.weight"], "storage_groups": ["storage.a", '
        '"storage.m", "storage.z"]}'
    )
