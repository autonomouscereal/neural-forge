"""Sealed module-boundary evidence and one global residual chart for native trunks.

``capture_hf_causal_circuit_evidence`` compiles a *layout*: it reads a
``DecoderLayout`` and rejects anything that is not llama/qwen3 shaped
(``hf_causal_circuit_evidence.py``).  A native trunk such as chimera-omni has
no such layout, so the promoted causal branch operators had no evidence bundle
to authenticate against and every chimera fold stayed driver-local
(``docs/CHIMERA_OMNI_FOLD_COVERAGE_2026-08-04.md`` section 3.3.2).

This module closes exactly that hole and nothing more.  It is deliberately
*graph driven* rather than family driven: the capture paths come from the
compiled :class:`~neural_forge.dynamic_circuit_ir.ModelCircuitGraph` -- every
module that owns a graph parameter, plus the module that owns each multi-tensor
operator node -- so the same capture works for any inspectable trunk whose IR
names its tensors.  Nothing here knows what a chimera block is.

What this bundle is, stated plainly:

* One forward, hooked, detached to float64 on CPU, sealed by a content digest
  over every captured tensor.  ``verify()`` recomputes the seal.
* A deterministic fit/validation split that follows the *token*, not the row,
  wherever a capture's leading two axes are the trunk's ``[batch, sequence]``.
  Captures on other axes (media segments) fall back to a modular row split;
  both are disjoint and both are named in the seal.
* One global orthonormal residual chart compiled from the residual rows the
  trunk actually observes.  This is *not* the production
  :class:`~neural_forge.hf_causal_cross_width_charts.ResidualBoundaryChartSet`:
  that artifact authenticates against a sealed ``HFCausalCircuitEvidence``
  bundle and carries per-boundary cycle/transition/absorption diagnostics.  One
  global chart makes every transition defect zero by construction and leaves
  the cycle and norm-commutation defects live; per-boundary charts for a
  ``block_attnres`` trunk are an explicitly named gap
  (``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 2).

This is not the per-family chimera evidence bundle the KDA/CSA/HCA design
reserves (``hf_chimera_circuit_evidence.py``); that one captures recurrence
state snapshots, retention rows and per-sub-branch deltas that no module
boundary exposes.  When it lands, this bundle stays the family-neutral module
boundary lane underneath it.
"""

from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .dynamic_circuit_ir import (
    CircuitKind,
    ModelCircuitGraph,
    PortKind,
    canonical_sha256,
)


EVIDENCE_SCHEMA = "neural-forge-native-circuit-evidence-v1"
CHART_SCHEMA = "neural-forge-native-global-residual-chart-v1"

#: One token in four is held out for validation and one for hyperparameter
#: selection, leaving half for the solves.  The number is a policy constant,
#: sealed in the bundle, not a tuning knob a caller may sweep per fit.
DEFAULT_FIT_PERIOD = 4

#: Knuth's 32-bit golden-ratio multiplier and the largest prime below 2**32.
#: Together they scatter the token index across partitions without resonating
#: with the sequence length, and they are integers, so the split is identical
#: on every platform and every PyTorch build.
_SPLIT_MULTIPLIER = 2654435761
_SPLIT_MODULUS = 4294967291


class NativeCircuitEvidenceError(ValueError):
    """A native evidence bundle or residual chart is invalid or unusable."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise NativeCircuitEvidenceError(
            "PyTorch is required for native circuit evidence."
        ) from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeCircuitEvidenceError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeCircuitEvidenceError(f"{name} must be a positive integer.")
    return value


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    state = hashlib.sha256()
    state.update(str(tuple(int(size) for size in tensor.shape)).encode("utf-8"))
    state.update(tensor.numpy().tobytes())
    return state.hexdigest()


def _cloned_rows(value: Any) -> Any:
    """Return one detached float64 CPU copy, or ``None`` for a non-row value."""

    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.dim() < 2:
        return None
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()


def owning_module_path(parameter_path: str) -> str:
    """Return the module that owns one graph parameter path."""

    module, _, _attribute = parameter_path.rpartition(".")
    if not module:
        raise NativeCircuitEvidenceError(
            f"Parameter path {parameter_path!r} does not name an owning module."
        )
    return module


_module_path = owning_module_path


def node_module_path(node: Any) -> str:
    """Return the module that owns one whole IR node.

    For a split-Q/K/V/O attention node this is the attention module itself,
    which is the only boundary where the branch delta appears as a value.
    """

    parameter_paths = tuple(node.parameter_paths)
    if not parameter_paths:
        raise NativeCircuitEvidenceError(
            f"Node {node.node_id!r} owns no tensor, so it has no module."
        )
    return _common_module_prefix(
        tuple(owning_module_path(value) for value in parameter_paths)
    )


def _common_module_prefix(paths: Sequence[str]) -> str:
    parts = [value.split(".") for value in paths]
    shared: list[str] = []
    for index in range(min(len(value) for value in parts)):
        candidate = parts[0][index]
        if any(value[index] != candidate for value in parts):
            break
        shared.append(candidate)
    if not shared:
        raise NativeCircuitEvidenceError(
            "Operator node tensors do not share an owning module prefix: "
            + ", ".join(sorted(paths))
        )
    return ".".join(shared)


def native_capture_paths(graph: ModelCircuitGraph) -> tuple[str, ...]:
    """Name every module boundary this bundle observes, from the graph alone.

    Two boundaries per parameterized node: the module that *owns* each tensor,
    and the module that owns the node as a whole.  For a split-Q/K/V/O
    attention node the second one is the attention module itself, which is the
    only place the branch delta appears as a value.
    """

    if not isinstance(graph, ModelCircuitGraph):
        raise NativeCircuitEvidenceError("graph must be a ModelCircuitGraph.")
    paths: set[str] = set()
    for node in graph.nodes:
        parameter_paths = tuple(node.parameter_paths)
        if not parameter_paths:
            continue
        owners = tuple(_module_path(value) for value in parameter_paths)
        paths.update(owners)
        paths.add(_common_module_prefix(owners))
    return tuple(sorted(paths))


class NativeModuleCapture:
    """One module's cached input and output rows from a single forward."""

    __slots__ = ("module_path", "call_count", "inputs", "output")

    def __init__(
        self,
        *,
        module_path: str,
        call_count: int,
        inputs: tuple[Any | None, ...],
        output: Any,
    ) -> None:
        if not isinstance(module_path, str) or not module_path:
            raise NativeCircuitEvidenceError("module_path must be a nonempty string.")
        self.module_path = module_path
        self.call_count = _positive_int(call_count, "call_count")
        self.inputs = tuple(inputs)
        self.output = output

    def identity(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "call_count": self.call_count,
            "inputs": [
                None
                if value is None
                else {
                    "shape": [int(size) for size in value.shape],
                    "sha256": _tensor_sha256(value),
                }
                for value in self.inputs
            ],
            "output": None
            if self.output is None
            else {
                "shape": [int(size) for size in self.output.shape],
                "sha256": _tensor_sha256(self.output),
            },
        }


class NativeCircuitEvidence:
    """A self-verifying immutable module-boundary cache for one trunk forward."""

    __slots__ = (
        "graph_fingerprint",
        "model_class",
        "batch_size",
        "sequence_length",
        "fit_period",
        "input_digest",
        "forward_calls",
        "metadata",
        "_captures",
        "_cache_key",
    )

    def __init__(
        self,
        *,
        graph_fingerprint: str,
        model_class: str,
        batch_size: int,
        sequence_length: int,
        fit_period: int,
        input_digest: str,
        captures: Sequence[NativeModuleCapture],
        metadata: Mapping[str, Any] | None = None,
        forward_calls: int = 1,
    ) -> None:
        self.graph_fingerprint = _digest(graph_fingerprint, "graph_fingerprint")
        if not isinstance(model_class, str) or not model_class:
            raise NativeCircuitEvidenceError("model_class must be a nonempty string.")
        self.model_class = model_class
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.sequence_length = _positive_int(sequence_length, "sequence_length")
        self.fit_period = _positive_int(fit_period, "fit_period")
        if self.fit_period < 3:
            raise NativeCircuitEvidenceError(
                "fit_period must leave a fit, a selection and a validation "
                "token in every period."
            )
        self.input_digest = _digest(input_digest, "input_digest")
        if forward_calls != 1:
            raise NativeCircuitEvidenceError(
                "A native evidence bundle must come from exactly one forward."
            )
        self.forward_calls = 1
        ordered = tuple(captures)
        if any(not isinstance(value, NativeModuleCapture) for value in ordered):
            raise NativeCircuitEvidenceError(
                "captures must all be NativeModuleCapture values."
            )
        paths = tuple(value.module_path for value in ordered)
        if len(set(paths)) != len(paths):
            raise NativeCircuitEvidenceError("Captured module paths must be unique.")
        self._captures = MappingProxyType(
            {value.module_path: value for value in ordered}
        )
        self.metadata = MappingProxyType(dict(metadata or {}))
        self._cache_key = canonical_sha256(self._body())
        self.verify()

    # -- identity -----------------------------------------------------------

    def _body(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "graph_fingerprint": self.graph_fingerprint,
            "model_class": self.model_class,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "fit_period": self.fit_period,
            "split_policy": self.split_policy,
            "input_digest": self.input_digest,
            "forward_calls": self.forward_calls,
            "metadata": dict(self.metadata),
            "captures": [
                self._captures[path].identity() for path in sorted(self._captures)
            ],
        }

    @property
    def split_policy(self) -> str:
        return (
            "hashed-token-index-fit-selection-validation-period-"
            f"{self.fit_period}"
            "-with-hashed-row-fallback-off-the-token-axis"
        )

    @property
    def cache_key(self) -> str:
        return self._cache_key

    @property
    def fingerprint(self) -> str:
        return self._cache_key

    @property
    def captured_module_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._captures))

    def verify(self) -> "NativeCircuitEvidence":
        if canonical_sha256(self._body()) != self._cache_key:
            raise NativeCircuitEvidenceError(
                "The native evidence bundle no longer matches its seal."
            )
        return self

    # -- access -------------------------------------------------------------

    def has(self, module_path: str) -> bool:
        return module_path in self._captures

    def capture(self, module_path: str) -> NativeModuleCapture:
        try:
            return self._captures[module_path]
        except KeyError as exc:
            raise NativeCircuitEvidenceError(
                f"No captured module boundary for {module_path!r}."
            ) from exc

    def output(self, module_path: str) -> Any:
        value = self.capture(module_path).output
        if value is None:
            raise NativeCircuitEvidenceError(
                f"Module {module_path!r} produced no captured row output."
            )
        return value

    def input(self, module_path: str, index: int = 0) -> Any:
        capture = self.capture(module_path)
        if index >= len(capture.inputs) or capture.inputs[index] is None:
            raise NativeCircuitEvidenceError(
                f"Module {module_path!r} has no captured input rows at {index}."
            )
        return capture.inputs[index]

    # -- splits -------------------------------------------------------------

    def _index_masks(self, count: int) -> tuple[Any, Any, Any]:
        """Split ``count`` ordered indices into fit / selection / validation.

        Three frozen partitions, not two.  A hyperparameter chosen on the same
        rows a solve was fitted on is not a held-out number, and a
        hyperparameter chosen on the validation rows destroys the only honest
        endpoint the fold has.  The selection partition exists so a ridge can
        be picked without touching either.
        """

        torch = _torch()
        index = torch.arange(int(count), dtype=torch.long)
        # A plain ``index % period`` aliases against the sequence length
        # whenever the period divides it, which quietly assigns whole *token
        # positions* to one partition -- and on a structured batch that makes
        # the held-out rows a different distribution rather than a held-out
        # sample.  The multiplicative hash is still deterministic and still
        # token-indexed; it just does not resonate with the sequence axis.
        residue = (index * _SPLIT_MULTIPLIER % _SPLIT_MODULUS) % self.fit_period
        validation = residue == (self.fit_period - 1)
        selection = residue == (self.fit_period - 2)
        return (~(validation | selection)), selection, validation

    def token_masks(self) -> tuple[Any, Any, Any, Any]:
        """Return ``(valid, fit, selection, validation)`` ``[batch, sequence]``."""

        torch = _torch()
        shape = (self.batch_size, self.sequence_length)
        fit, selection, validation = self._index_masks(
            self.batch_size * self.sequence_length
        )
        return (
            torch.ones(shape, dtype=torch.bool),
            fit.reshape(shape).contiguous(),
            selection.reshape(shape).contiguous(),
            validation.reshape(shape).contiguous(),
        )

    def row_split(self, rows: Any) -> tuple[Any, Any, Any]:
        """Return disjoint ``(fit, selection, validation)`` masks for one capture.

        A capture whose two leading axes are the trunk's ``[batch, sequence]``
        inherits the *token* split, so a token held out of one fit is held out
        of every fit that sees it.  Anything else -- media segment rows, depth
        source stacks flattened past the token axis -- keeps the same period on
        its own row index and says so.
        """

        torch = _torch()
        if not isinstance(rows, torch.Tensor) or rows.dim() < 2:
            raise NativeCircuitEvidenceError("row_split requires a rank>=2 tensor.")
        shape = tuple(int(size) for size in rows.shape)
        if shape[:2] == (self.batch_size, self.sequence_length):
            _valid, fit, selection, validation = self.token_masks()
            repeats = 1
            for size in shape[2:-1]:
                repeats *= int(size)
            masks = tuple(
                value.reshape(-1, 1).expand(-1, repeats).reshape(-1).contiguous()
                for value in (fit, selection, validation)
            )
        else:
            count = 1
            for size in shape[:-1]:
                count *= int(size)
            masks = tuple(
                value.contiguous() for value in self._index_masks(count)
            )
        if any(not bool(value.any().item()) for value in masks):
            raise NativeCircuitEvidenceError(
                "A captured row set is too small to hold out any evidence."
            )
        return masks

    def flat_rows(self, value: Any) -> Any:
        torch = _torch()
        if not isinstance(value, torch.Tensor) or value.dim() < 2:
            raise NativeCircuitEvidenceError("flat_rows requires a rank>=2 tensor.")
        return value.reshape(-1, int(value.shape[-1])).contiguous()


def capture_native_circuit_evidence(
    model: Any,
    graph: ModelCircuitGraph,
    *,
    inputs: Mapping[str, Any],
    forward: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    sequence_key: str = "input_ids",
    fit_period: int = DEFAULT_FIT_PERIOD,
    metadata: Mapping[str, Any] | None = None,
) -> NativeCircuitEvidence:
    """Capture one sealed module-boundary bundle from exactly one forward."""

    torch = _torch()
    if not isinstance(graph, ModelCircuitGraph):
        raise NativeCircuitEvidenceError("graph must be a ModelCircuitGraph.")
    if not isinstance(inputs, Mapping) or sequence_key not in inputs:
        raise NativeCircuitEvidenceError(
            f"inputs must be a mapping containing {sequence_key!r}."
        )
    sequences = inputs[sequence_key]
    if not isinstance(sequences, torch.Tensor) or sequences.dim() != 2:
        raise NativeCircuitEvidenceError(
            f"inputs[{sequence_key!r}] must be a [batch, sequence] tensor."
        )
    paths = native_capture_paths(graph)
    records: dict[str, dict[str, Any]] = {}

    def make_hook(path: str) -> Callable[..., None]:
        def hook(_module: Any, arguments: Any, output: Any) -> None:
            record = records.setdefault(path, {"calls": 0, "inputs": (), "output": None})
            record["calls"] += 1
            if record["calls"] > 1:
                # A module invoked twice in one forward would make "the" cached
                # row set ambiguous; the first call is kept and the count is
                # sealed so the operator can refuse the boundary.
                return
            record["inputs"] = tuple(_cloned_rows(value) for value in arguments)
            first = output[0] if isinstance(output, tuple) else output
            record["output"] = _cloned_rows(first)

        return hook

    handles = []
    try:
        for path in paths:
            try:
                module = model.get_submodule(path)
            except AttributeError as exc:
                raise NativeCircuitEvidenceError(
                    f"The model does not expose the graph module {path!r}."
                ) from exc
            handles.append(module.register_forward_hook(make_hook(path)))
        caller = forward if forward is not None else _default_forward
        with torch.no_grad():
            caller(model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    captures = tuple(
        NativeModuleCapture(
            module_path=path,
            call_count=int(record["calls"]),
            inputs=tuple(record["inputs"]),
            output=record["output"],
        )
        for path, record in sorted(records.items())
    )
    digest = canonical_sha256(
        {
            "kind": "native-circuit-evidence-inputs",
            "tensors": [
                [
                    name,
                    [int(size) for size in inputs[name].shape],
                    _tensor_sha256(inputs[name]),
                ]
                for name in sorted(inputs)
                if isinstance(inputs[name], torch.Tensor)
            ],
        }
    )
    return NativeCircuitEvidence(
        graph_fingerprint=graph.fingerprint,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        batch_size=int(sequences.shape[0]),
        sequence_length=int(sequences.shape[1]),
        fit_period=fit_period,
        input_digest=digest,
        captures=captures,
        metadata=metadata,
    )


def _default_forward(model: Any, inputs: Mapping[str, Any]) -> Any:
    return model(**inputs)


class NativeChildRecapture:
    """Recapture the live child at every module boundary the graph names.

    The parent bundle is sealed and never regenerated; the *child* is a moving
    target during a fold, so every operator that fits against child rows must
    read the child as it is at that moment.  This is the native analogue of
    ``LiveChildBoundaryRecapture``.
    """

    __slots__ = ("inputs", "forward", "sequence_key", "fit_period", "calls")

    def __init__(
        self,
        inputs: Mapping[str, Any],
        *,
        forward: Callable[[Any, Mapping[str, Any]], Any] | None = None,
        sequence_key: str = "input_ids",
        fit_period: int = DEFAULT_FIT_PERIOD,
    ) -> None:
        self.inputs = dict(inputs)
        self.forward = forward
        self.sequence_key = sequence_key
        self.fit_period = _positive_int(fit_period, "fit_period")
        self.calls = 0

    @property
    def identity(self) -> str:
        return canonical_sha256(
            {
                "kind": "native-child-recapture",
                "sequence_key": self.sequence_key,
                "fit_period": self.fit_period,
                "input_names": sorted(self.inputs),
            }
        )

    def __call__(
        self,
        model: Any,
        graph: ModelCircuitGraph,
    ) -> NativeCircuitEvidence:
        self.calls += 1
        return capture_native_circuit_evidence(
            model,
            graph,
            inputs=self.inputs,
            forward=self.forward,
            sequence_key=self.sequence_key,
            fit_period=self.fit_period,
            metadata={"role": "live_child_recapture"},
        )


def residual_chart_pool_paths(graph: ModelCircuitGraph) -> tuple[str, ...]:
    """Name the module boundaries whose inputs are residual-coordinate rows.

    Two classes, both read off the graph: a ``NORM`` node whose input port is
    a residual, and the internal normalization of a parameterized
    ``RESIDUAL_ADD`` depth selector, whose input is the stack of depth sources
    the selector really compared.  Nothing else is pooled -- media-lane rows
    enter the residual stream only after their own projection, so pooling them
    would let a boundary the trunk never observes vote on the chart.
    """

    if not isinstance(graph, ModelCircuitGraph):
        raise NativeCircuitEvidenceError("graph must be a ModelCircuitGraph.")
    paths: list[str] = []
    for node in graph.nodes:
        if node.kind == CircuitKind.NORM.value:
            if any(port.kind == PortKind.RESIDUAL.value for port in node.inputs):
                paths.extend(
                    _module_path(value) for value in tuple(node.parameter_paths)[:1]
                )
            continue
        if node.kind != CircuitKind.RESIDUAL_ADD.value:
            continue
        for parameter in node.parameters:
            if "norm" in str(parameter.role).lower():
                paths.append(_module_path(parameter.path))
    return tuple(sorted(set(paths)))


class NativeResidualChart:
    """One global orthonormal residual chart with an explicit contraction seal."""

    __slots__ = (
        "encoder",
        "source_width",
        "target_width",
        "map_sha256",
        "_report",
    )

    def __init__(self, encoder: Any, report: Mapping[str, Any]) -> None:
        torch = _torch()
        if not isinstance(encoder, torch.Tensor) or encoder.dim() != 2:
            raise NativeCircuitEvidenceError("A residual chart encoder is a matrix.")
        matrix = encoder.detach().to(device="cpu", dtype=torch.float64).contiguous()
        source_width = int(matrix.shape[0])
        target_width = int(matrix.shape[1])
        if target_width > source_width:
            raise NativeCircuitEvidenceError(
                "A residual chart is contraction-only; the target width exceeds "
                "the source width."
            )
        defect = float(
            (
                matrix.transpose(0, 1) @ matrix
                - torch.eye(target_width, dtype=torch.float64)
            )
            .abs()
            .max()
            .item()
        )
        if not (defect <= 1.0e-9):
            raise NativeCircuitEvidenceError(
                f"The residual chart is not orthonormal: defect {defect!r}."
            )
        self.encoder = matrix
        self.source_width = source_width
        self.target_width = target_width
        self.map_sha256 = _tensor_sha256(matrix)
        self._report = MappingProxyType(
            {
                **dict(report),
                "schema": CHART_SCHEMA,
                "source_width": source_width,
                "target_width": target_width,
                "orthonormality_defect": defect,
                "encoder_sha256": self.map_sha256,
                "per_boundary_chart_set": False,
                "transition_defects_zero_by_construction": True,
            }
        )

    @property
    def report(self) -> Mapping[str, Any]:
        return self._report

    def encode(self, rows: Any) -> Any:
        """Project residual rows into child coordinates (``rows @ E``)."""

        torch = _torch()
        if not isinstance(rows, torch.Tensor) or rows.dim() < 1:
            raise NativeCircuitEvidenceError("encode requires a tensor.")
        if int(rows.shape[-1]) != self.source_width:
            raise NativeCircuitEvidenceError(
                "encode received rows of the wrong residual width."
            )
        return (rows.to(dtype=torch.float64) @ self.encoder).contiguous()

    def fold_consumer_weight(self, weight: Any) -> Any:
        """Fold a weight that *reads* the residual (``W E``)."""

        torch = _torch()
        matrix = weight.detach().to(device="cpu", dtype=torch.float64)
        if matrix.dim() != 2 or int(matrix.shape[1]) != self.source_width:
            raise NativeCircuitEvidenceError(
                "fold_consumer_weight requires a [out, source_width] matrix."
            )
        return (matrix @ self.encoder).contiguous()

    def fold_producer_weight(self, weight: Any) -> Any:
        """Fold a weight that *writes* the residual (``E^T W``)."""

        torch = _torch()
        matrix = weight.detach().to(device="cpu", dtype=torch.float64)
        if matrix.dim() != 2 or int(matrix.shape[0]) != self.source_width:
            raise NativeCircuitEvidenceError(
                "fold_producer_weight requires a [source_width, in] matrix."
            )
        return (self.encoder.transpose(0, 1) @ matrix).contiguous()

    def verify(self) -> "NativeResidualChart":
        if _tensor_sha256(self.encoder) != self.map_sha256:
            raise NativeCircuitEvidenceError(
                "The residual chart encoder no longer matches its seal."
            )
        return self


def compile_native_residual_chart(
    evidence: NativeCircuitEvidence,
    graph: ModelCircuitGraph,
    *,
    target_width: int,
) -> NativeResidualChart:
    """Compile the leading residual subspace of one sealed parent bundle."""

    torch = _torch()
    if not isinstance(evidence, NativeCircuitEvidence):
        raise NativeCircuitEvidenceError("evidence must be NativeCircuitEvidence.")
    evidence.verify()
    if evidence.graph_fingerprint != graph.fingerprint:
        raise NativeCircuitEvidenceError(
            "The residual chart evidence is bound to a different graph."
        )
    target_width = _positive_int(target_width, "target_width")
    pool: list[Any] = []
    sources: list[str] = []
    for path in residual_chart_pool_paths(graph):
        if not evidence.has(path):
            continue
        capture = evidence.capture(path)
        if not capture.inputs or capture.inputs[0] is None:
            continue
        pool.append(evidence.flat_rows(capture.inputs[0]))
        sources.append(path)
    if not pool:
        raise NativeCircuitEvidenceError(
            "No residual-coordinate rows were captured for a chart."
        )
    widths = {int(value.shape[1]) for value in pool}
    if len(widths) != 1:
        raise NativeCircuitEvidenceError(
            "Pooled residual rows do not share one residual width."
        )
    rows = torch.cat(pool, dim=0)
    _left, singular_values, right = torch.linalg.svd(rows, full_matrices=False)
    if target_width > int(right.shape[0]):
        raise NativeCircuitEvidenceError(
            "The requested chart rank exceeds the observed residual rank."
        )
    encoder = right[:target_width].transpose(0, 1).contiguous()
    energy = singular_values.square()
    total = float(energy.sum().item())
    retained = float(energy[:target_width].sum().item())
    projection = (rows @ encoder) @ encoder.transpose(0, 1)
    error = float((projection - rows).square().mean().sqrt().item())
    reference = float(rows.square().mean().sqrt().item())
    return NativeResidualChart(
        encoder,
        {
            "method": "single_global_orthonormal_residual_chart",
            "parent_evidence_sha256": evidence.cache_key,
            "pooled_rows": int(rows.shape[0]),
            "pooled_sources": list(sources),
            "retained_energy_fraction": retained / max(total, 1.0e-300),
            "projection_relative_rmse": error / max(reference, 1.0e-300),
            "per_boundary_chart_set_gap": (
                "ResidualBoundaryChartSet authenticates against a sealed "
                "HFCausalCircuitEvidence bundle and carries per-boundary "
                "cycle/transition/absorption diagnostics; a native trunk "
                "cannot produce that bundle, so one global chart is used and "
                "the per-boundary path stays a named gap."
            ),
        },
    )


__all__ = [
    "CHART_SCHEMA",
    "DEFAULT_FIT_PERIOD",
    "EVIDENCE_SCHEMA",
    "NativeChildRecapture",
    "NativeCircuitEvidence",
    "NativeCircuitEvidenceError",
    "NativeModuleCapture",
    "NativeResidualChart",
    "capture_native_circuit_evidence",
    "compile_native_residual_chart",
    "native_capture_paths",
    "node_module_path",
    "owning_module_path",
    "residual_chart_pool_paths",
]
