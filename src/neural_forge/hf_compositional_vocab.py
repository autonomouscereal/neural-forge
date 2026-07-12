"""Trainable tied vocabulary weights represented by product medoid codebooks.

The hidden axis is split into deterministic contiguous subspaces.  Each token
selects one real source-row medoid in every subspace, so the logical token row
is the concatenation of those selected codewords.  Runtime embedding and LM
head views share one registry; neither view owns duplicate parameters.

The LM head computes subspace/codeword dot products and gathers them through
the token assignments.  It therefore produces the unavoidable vocabulary
logits without constructing a dense ``vocab_size x hidden_size`` weight.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_FORMAT_VERSION = 1
_INITIALIZATION_NAMES = {
    0: "manual_codebooks",
    1: "supplied_assignments",
    2: "hash_minibatch_farthest_first",
}
_INITIALIZATION_CODES = {value: key for key, value in _INITIALIZATION_NAMES.items()}
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}

# Integer slots in the tensor-only ``_fit_metadata`` buffer.
_META_INITIALIZATION = 0
_META_SEED = 1
_META_MAX_ITERATIONS = 2
_META_ITERATIONS_RUN = 3
_META_REQUESTED_TOKEN_CHUNK = 4
_META_REQUESTED_CODEWORD_CHUNK = 5
_META_REQUESTED_MINIBATCH = 6
_META_DISTANCE_VALUES = 7
_META_MAX_DISTANCE_BLOCK = 8
_META_SUPPLIED_ASSIGNMENTS = 9
_FIT_METADATA_LENGTH = 10


class CompositionalVocabularyError(ValueError):
    """Base class for compositional vocabulary errors."""


class CompositionalVocabularyShapeError(CompositionalVocabularyError):
    """A source, codebook, assignment, or runtime input has an invalid shape."""


class CompositionalVocabularyFiniteError(CompositionalVocabularyError):
    """A floating-point input contains NaN or infinity."""


class CompositionalVocabularyStateError(CompositionalVocabularyError):
    """A tensor state mapping cannot reconstruct a vocabulary registry."""


@dataclass(frozen=True, slots=True)
class CompositionalVocabularyReport:
    """Separate learned, index, dense-equivalent, and serialized accounting."""

    vocab_size: int
    hidden_size: int
    subspace_count: int
    subspace_widths: tuple[int, ...]
    codebook_sizes: tuple[int, ...]
    codebook_parameters: int
    bias_parameters: int
    learned_parameter_count: int
    learned_parameter_bytes: int
    assignment_index_bytes: int
    medoid_source_index_bytes: int
    metadata_index_bytes: int
    index_buffer_bytes: int
    other_tensor_buffer_bytes: int
    tensor_payload_bytes: int
    dense_weight_parameter_count: int
    dense_equivalent_parameter_count: int
    dense_equivalent_bytes: int
    actual_serialized_bytes: int

    @property
    def learned_parameters(self) -> int:
        return self.learned_parameter_count

    @property
    def unique_learned_parameters(self) -> int:
        return self.learned_parameter_count

    @property
    def unique_parameter_count(self) -> int:
        return self.learned_parameter_count

    @property
    def assignment_bytes(self) -> int:
        return self.assignment_index_bytes

    @property
    def index_bytes(self) -> int:
        return self.index_buffer_bytes

    @property
    def dense_equivalent_count(self) -> int:
        return self.dense_equivalent_parameter_count

    @property
    def serialized_bytes(self) -> int:
        return self.actual_serialized_bytes

    @property
    def learned_to_dense_ratio(self) -> float:
        return self.learned_parameter_count / self.dense_equivalent_parameter_count

    @property
    def serialized_to_dense_ratio(self) -> float:
        return self.actual_serialized_bytes / self.dense_equivalent_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "subspace_count": self.subspace_count,
            "subspace_widths": list(self.subspace_widths),
            "codebook_sizes": list(self.codebook_sizes),
            "codebook_parameters": self.codebook_parameters,
            "bias_parameters": self.bias_parameters,
            "learned_parameter_count": self.learned_parameter_count,
            "learned_parameters": self.learned_parameter_count,
            "unique_learned_parameters": self.learned_parameter_count,
            "learned_parameter_bytes": self.learned_parameter_bytes,
            "assignment_index_bytes": self.assignment_index_bytes,
            "assignment_bytes": self.assignment_index_bytes,
            "medoid_source_index_bytes": self.medoid_source_index_bytes,
            "metadata_index_bytes": self.metadata_index_bytes,
            "index_buffer_bytes": self.index_buffer_bytes,
            "index_bytes": self.index_buffer_bytes,
            "other_tensor_buffer_bytes": self.other_tensor_buffer_bytes,
            "tensor_payload_bytes": self.tensor_payload_bytes,
            "dense_weight_parameter_count": self.dense_weight_parameter_count,
            "dense_equivalent_parameter_count": (self.dense_equivalent_parameter_count),
            "dense_equivalent_count": self.dense_equivalent_parameter_count,
            "dense_equivalent_bytes": self.dense_equivalent_bytes,
            "actual_serialized_bytes": self.actual_serialized_bytes,
            "serialized_bytes": self.actual_serialized_bytes,
            "learned_to_dense_ratio": self.learned_to_dense_ratio,
            "serialized_to_dense_ratio": self.serialized_to_dense_ratio,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class CompositionalVocabularyFitDiagnostics:
    """Evidence that fitting is deterministic, bounded, and matrix-scalable."""

    algorithm: str
    initialization: str
    deterministic: bool
    supplied_assignments: bool
    assignments_refined: bool
    real_medoid_codewords: bool
    tie_break: str
    vocab_size: int
    hidden_size: int
    subspace_count: int
    subspace_widths: tuple[int, ...]
    codebook_sizes: tuple[int, ...]
    seed: int
    requested_initialization_minibatch_size: int
    initialization_sample_sizes: tuple[int, ...]
    requested_token_chunk_size: int
    requested_codeword_chunk_size: int
    effective_token_chunk_sizes: tuple[int, ...]
    effective_codeword_chunk_sizes: tuple[int, ...]
    max_iterations: int
    iterations_per_subspace: tuple[int, ...]
    iterations_run: int
    distance_values_computed: int
    distance_work_bound: int
    max_distance_block_elements: int
    distance_block_element_limit: int
    full_vocabulary_by_codebook_matrix_elements: int
    largest_vocabulary_by_codebook_matrix_elements: int
    materialized_full_vocabulary_by_codebook_matrix: bool
    bounded_work: bool
    fitting_requires_grad: bool
    squared_reconstruction_error: float

    @property
    def no_full_vocabulary_by_codebook_matrix(self) -> bool:
        return not self.materialized_full_vocabulary_by_codebook_matrix

    @property
    def no_full_assignment_matrix(self) -> bool:
        return self.no_full_vocabulary_by_codebook_matrix

    @property
    def materialized_v_by_k_matrix(self) -> bool:
        return self.materialized_full_vocabulary_by_codebook_matrix

    @property
    def real_medoids(self) -> bool:
        return self.real_medoid_codewords

    def as_dict(self) -> dict[str, Any]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["no_full_vocabulary_by_codebook_matrix"] = (
            self.no_full_vocabulary_by_codebook_matrix
        )
        result["no_full_assignment_matrix"] = self.no_full_assignment_matrix
        return result

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(slots=True)
class _WorkTracker:
    distance_values: int = 0
    max_block_elements: int = 0

    def add(self, rows: int, columns: int) -> None:
        elements = rows * columns
        self.distance_values += elements
        self.max_block_elements = max(self.max_block_elements, elements)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CompositionalVocabularyShapeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompositionalVocabularyShapeError(
            f"{name} must be a non-negative integer."
        )
    return value


def _tensor_bytes(value: Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def serialized_tensor_state_bytes(state_dict: Mapping[str, Tensor]) -> int:
    """Return the exact bytes emitted by ``torch.save`` for this state mapping."""

    if not isinstance(state_dict, Mapping):
        raise CompositionalVocabularyStateError(
            "state_dict must be a mapping of names to tensors."
        )
    if any(not isinstance(value, Tensor) for value in state_dict.values()):
        raise CompositionalVocabularyStateError(
            "state_dict may contain tensor values only."
        )
    stream = io.BytesIO()
    torch.save(state_dict, stream)
    return stream.tell()


def split_hidden_subspaces(
    hidden_size: int,
    subspace_count: int,
) -> tuple[tuple[int, int], ...]:
    """Split a hidden axis into stable contiguous widths differing by at most one."""

    hidden_size = _positive_int(hidden_size, "hidden_size")
    subspace_count = _positive_int(subspace_count, "subspace_count")
    if subspace_count > hidden_size:
        raise CompositionalVocabularyShapeError(
            "subspace_count cannot exceed hidden_size."
        )
    base, remainder = divmod(hidden_size, subspace_count)
    result: list[tuple[int, int]] = []
    start = 0
    for index in range(subspace_count):
        width = base + (1 if index < remainder else 0)
        result.append((start, start + width))
        start += width
    return tuple(result)


def _validate_float_tensor(value: Any, name: str, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise CompositionalVocabularyShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise CompositionalVocabularyShapeError(
            f"{name} must be a dense non-meta tensor."
        )
    if value.ndim != dimensions or any(int(size) <= 0 for size in value.shape):
        raise CompositionalVocabularyShapeError(
            f"{name} must have {dimensions} non-empty dimensions."
        )
    if not torch.is_floating_point(value):
        raise CompositionalVocabularyShapeError(
            f"{name} must use a real floating-point dtype."
        )
    if not bool(torch.isfinite(value.detach()).all()):
        raise CompositionalVocabularyFiniteError(f"{name} contains NaN or infinity.")
    return value


def _validate_integer_matrix(value: Any, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype not in _INTEGER_DTYPES
    ):
        raise CompositionalVocabularyShapeError(
            f"{name} must be a non-empty dense integer matrix."
        )
    return value


def _same_tensor_storage(left: Tensor, right: Tensor) -> bool:
    if left is right:
        return True
    if (
        left.device != right.device
        or left.dtype != right.dtype
        or left.shape != right.shape
        or left.stride() != right.stride()
        or left.storage_offset() != right.storage_offset()
    ):
        return False
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


class CompositionalVocabularyRegistry(nn.Module):
    """The sole registered owner of trainable codebooks and index buffers."""

    lm_head_bias: nn.Parameter | None

    def __init__(
        self,
        codebooks: Sequence[Tensor],
        assignments: Tensor,
        medoid_source_indices: Sequence[Tensor],
        *,
        subspace_offsets: Tensor | Sequence[int] | None = None,
        lm_head_bias: Tensor | None = None,
        padding_idx: int | None = None,
        fit_metadata: Tensor | None = None,
        iterations_per_subspace: Tensor | None = None,
        initialization_sample_sizes: Tensor | None = None,
        effective_token_chunk_sizes: Tensor | None = None,
        effective_codeword_chunk_sizes: Tensor | None = None,
        fit_squared_error: Tensor | float | None = None,
    ) -> None:
        super().__init__()
        if isinstance(codebooks, (str, bytes)) or not isinstance(codebooks, Sequence):
            raise CompositionalVocabularyShapeError(
                "codebooks must be a non-empty tensor sequence."
            )
        codebook_values = tuple(codebooks)
        if not codebook_values:
            raise CompositionalVocabularyShapeError("codebooks cannot be empty.")
        checked = tuple(
            _validate_float_tensor(value, f"codebooks[{index}]", 2)
            for index, value in enumerate(codebook_values)
        )
        reference = checked[0]
        if any(
            value.dtype != reference.dtype or value.device != reference.device
            for value in checked[1:]
        ):
            raise CompositionalVocabularyShapeError(
                "all codebooks must have the same dtype and device."
            )

        assignments = _validate_integer_matrix(assignments, "assignments")
        vocab_size, subspace_count = map(int, assignments.shape)
        if subspace_count != len(checked):
            raise CompositionalVocabularyShapeError(
                "assignments columns must equal the number of codebooks."
            )
        if len(medoid_source_indices) != subspace_count:
            raise CompositionalVocabularyShapeError(
                "medoid_source_indices must contain one tensor per subspace."
            )

        widths = tuple(int(value.shape[1]) for value in checked)
        if subspace_offsets is None:
            offsets = [0]
            for width in widths:
                offsets.append(offsets[-1] + width)
            offsets_tensor = torch.tensor(offsets, dtype=torch.int64)
        else:
            offsets_tensor = torch.as_tensor(subspace_offsets).detach().to("cpu")
            if offsets_tensor.dtype not in _INTEGER_DTYPES:
                raise CompositionalVocabularyShapeError(
                    "subspace_offsets must use an integer dtype."
                )
            offsets_tensor = offsets_tensor.to(torch.int64)
        expected_offsets = [0]
        for width in widths:
            expected_offsets.append(expected_offsets[-1] + width)
        if offsets_tensor.ndim != 1 or offsets_tensor.tolist() != expected_offsets:
            raise CompositionalVocabularyShapeError(
                "subspace_offsets must exactly match the codebook widths."
            )

        assignment64 = assignments.detach().to(device="cpu", dtype=torch.int64)
        medoid_values: list[Tensor] = []
        codebook_sizes: list[int] = []
        for index, (codebook, medoids) in enumerate(
            zip(checked, medoid_source_indices, strict=True)
        ):
            codebook_size = int(codebook.shape[0])
            codebook_sizes.append(codebook_size)
            column = assignment64[:, index]
            if int(column.min()) < 0 or int(column.max()) >= codebook_size:
                raise CompositionalVocabularyShapeError(
                    f"assignments column {index} has an out-of-range codeword."
                )
            if (
                not isinstance(medoids, Tensor)
                or medoids.layout != torch.strided
                or medoids.device.type == "meta"
                or medoids.ndim != 1
                or int(medoids.numel()) != codebook_size
                or medoids.dtype not in _INTEGER_DTYPES
            ):
                raise CompositionalVocabularyShapeError(
                    "each medoid source tensor must have one integer per codeword."
                )
            medoid64 = medoids.detach().to(device="cpu", dtype=torch.int64)
            if int(medoid64.min()) < 0 or int(medoid64.max()) >= vocab_size:
                raise CompositionalVocabularyShapeError(
                    f"medoid_source_indices[{index}] is outside the vocabulary."
                )
            if int(torch.unique(medoid64).numel()) != codebook_size:
                raise CompositionalVocabularyShapeError(
                    "medoid source rows must be distinct within each subspace."
                )
            medoid_values.append(medoids)

        if vocab_size > torch.iinfo(torch.int32).max:
            raise CompositionalVocabularyShapeError(
                "vocab_size exceeds the supported int32 assignment index range."
            )
        if max(codebook_sizes) > torch.iinfo(torch.int32).max:
            raise CompositionalVocabularyShapeError(
                "codebook size exceeds the supported int32 assignment range."
            )
        if padding_idx is not None:
            if isinstance(padding_idx, bool) or not isinstance(padding_idx, int):
                raise CompositionalVocabularyShapeError(
                    "padding_idx must be an integer or None."
                )
            if padding_idx < 0:
                padding_idx += vocab_size
            if not 0 <= padding_idx < vocab_size:
                raise CompositionalVocabularyShapeError(
                    "padding_idx is outside the vocabulary."
                )

        self.codebooks = nn.ParameterList(
            nn.Parameter(value.detach().clone().contiguous(), requires_grad=True)
            for value in checked
        )
        device = reference.device
        self.register_buffer(
            "assignments",
            assignments.detach().to(device=device, dtype=torch.int32).clone(),
        )
        self.register_buffer(
            "subspace_offsets",
            offsets_tensor.to(device=device, dtype=torch.int64),
        )
        self.register_buffer(
            "codebook_sizes",
            torch.tensor(codebook_sizes, device=device, dtype=torch.int64),
        )
        for index, value in enumerate(medoid_values):
            self.register_buffer(
                f"medoid_source_indices_{index}",
                value.detach().to(device=device, dtype=torch.int32).clone(),
            )
        self.register_buffer(
            "_format_version",
            torch.tensor(_FORMAT_VERSION, device=device, dtype=torch.int64),
        )
        self.register_buffer(
            "_padding_idx",
            torch.tensor(
                -1 if padding_idx is None else padding_idx,
                device=device,
                dtype=torch.int64,
            ),
        )

        if lm_head_bias is None:
            self.register_parameter("lm_head_bias", None)
        else:
            bias = _validate_float_tensor(lm_head_bias, "lm_head_bias", 1)
            if int(bias.numel()) != vocab_size:
                raise CompositionalVocabularyShapeError(
                    "lm_head_bias must contain one value per token."
                )
            if bias.dtype != reference.dtype or bias.device != reference.device:
                raise CompositionalVocabularyShapeError(
                    "lm_head_bias must match the codebook dtype and device."
                )
            self.lm_head_bias = nn.Parameter(
                bias.detach().clone().contiguous(), requires_grad=True
            )

        metadata = self._fit_tensor(
            fit_metadata,
            default=torch.tensor(
                [
                    _INITIALIZATION_CODES["manual_codebooks"],
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=torch.int64,
            ),
            name="fit_metadata",
            shape=(_FIT_METADATA_LENGTH,),
            dtype=torch.int64,
            device=device,
        )
        self.register_buffer("_fit_metadata", metadata)
        for name, supplied in (
            ("_iterations_per_subspace", iterations_per_subspace),
            ("_initialization_sample_sizes", initialization_sample_sizes),
            ("_effective_token_chunk_sizes", effective_token_chunk_sizes),
            ("_effective_codeword_chunk_sizes", effective_codeword_chunk_sizes),
        ):
            default = torch.zeros(subspace_count, dtype=torch.int64)
            if name.startswith("_effective"):
                default.fill_(1)
            tensor = self._fit_tensor(
                supplied,
                default=default,
                name=name.removeprefix("_"),
                shape=(subspace_count,),
                dtype=torch.int64,
                device=device,
            )
            self.register_buffer(name, tensor)
        error_value = (
            torch.tensor(0.0, dtype=torch.float64)
            if fit_squared_error is None
            else torch.as_tensor(fit_squared_error)
        )
        error = self._fit_tensor(
            error_value,
            default=torch.tensor(0.0, dtype=torch.float64),
            name="fit_squared_error",
            shape=(),
            dtype=torch.float64,
            device=device,
        )
        if not bool(torch.isfinite(error).all()):
            raise CompositionalVocabularyFiniteError(
                "fit_squared_error must be finite."
            )
        self.register_buffer("_fit_squared_error", error)

    @staticmethod
    def _fit_tensor(
        value: Tensor | None,
        *,
        default: Tensor,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        tensor = default if value is None else value
        if not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape:
            raise CompositionalVocabularyShapeError(
                f"{name} must be a tensor with shape {shape}."
            )
        if dtype == torch.int64 and tensor.dtype not in _INTEGER_DTYPES:
            raise CompositionalVocabularyShapeError(
                f"{name} must use an integer dtype."
            )
        if dtype == torch.float64 and not torch.is_floating_point(tensor):
            raise CompositionalVocabularyShapeError(
                f"{name} must use a floating-point dtype."
            )
        return tensor.detach().to(device=device, dtype=dtype).clone()

    @property
    def vocab_size(self) -> int:
        return int(self.assignments.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.subspace_offsets[-1].item())

    @property
    def subspace_count(self) -> int:
        return len(self.codebooks)

    @property
    def subspace_widths(self) -> tuple[int, ...]:
        return tuple(int(value.shape[1]) for value in self.codebooks)

    @property
    def padding_idx(self) -> int | None:
        value = int(self._padding_idx.item())
        return None if value < 0 else value

    @property
    def medoid_source_indices(self) -> tuple[Tensor, ...]:
        return tuple(
            getattr(self, f"medoid_source_indices_{index}")
            for index in range(self.subspace_count)
        )

    def _subspace_index(self, index: int) -> int:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.subspace_count
        ):
            raise CompositionalVocabularyShapeError(
                "subspace index is outside the codebook registry."
            )
        return index

    def subspace_slice(self, index: int) -> slice:
        index = self._subspace_index(index)
        return slice(
            int(self.subspace_offsets[index].item()),
            int(self.subspace_offsets[index + 1].item()),
        )

    def materialize_weight(self) -> Tensor:
        """Explicitly reconstruct the logical dense weight when requested."""

        parts = []
        for index, codebook in enumerate(self.codebooks):
            token_codewords = self.assignments[:, index]
            parts.append(codebook.index_select(0, token_codewords))
        return torch.cat(parts, dim=1).contiguous()

    reconstruct_weight = materialize_weight

    def embed(self, input_ids: Tensor) -> Tensor:
        if not isinstance(input_ids, Tensor) or input_ids.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise CompositionalVocabularyShapeError(
                "input_ids must be an int32 or int64 tensor."
            )
        parts = []
        for index, codebook in enumerate(self.codebooks):
            token_assignments = self.assignments[:, index]
            flat_codeword_ids = token_assignments.index_select(0, input_ids.reshape(-1))
            codeword_ids = flat_codeword_ids.reshape(input_ids.shape)
            parts.append(F.embedding(codeword_ids, codebook))
        output = torch.cat(parts, dim=-1)
        if self.padding_idx is not None:
            padding = input_ids.eq(self.padding_idx).unsqueeze(-1)
            output = torch.where(padding, output.detach(), output)
        return output

    def logits(self, hidden_states: Tensor) -> Tensor:
        if not isinstance(hidden_states, Tensor) or hidden_states.ndim == 0:
            raise CompositionalVocabularyShapeError(
                "hidden_states must be a non-scalar tensor."
            )
        if int(hidden_states.shape[-1]) != self.hidden_size:
            raise CompositionalVocabularyShapeError(
                f"hidden_states must end in width {self.hidden_size}."
            )
        logits: Tensor | None = None
        for index, codebook in enumerate(self.codebooks):
            subspace_scores = F.linear(
                hidden_states[..., self.subspace_slice(index)], codebook
            )
            gathered = subspace_scores.index_select(-1, self.assignments[:, index])
            logits = gathered if logits is None else logits + gathered
        if logits is None:  # pragma: no cover - construction forbids no codebooks
            raise RuntimeError("A vocabulary registry must contain a codebook.")
        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias
        return logits

    def forward(self, inputs: Tensor, *, operation: str) -> Tensor:
        if operation == "embedding":
            return self.embed(inputs)
        if operation == "lm_head":
            return self.logits(inputs)
        raise CompositionalVocabularyShapeError(
            "operation must be 'embedding' or 'lm_head'."
        )

    def _report_for_state(
        self,
        state_dict: Mapping[str, Tensor],
    ) -> CompositionalVocabularyReport:
        parameter_values = tuple(self.parameters())
        learned_count = sum(int(value.numel()) for value in parameter_values)
        learned_bytes = sum(_tensor_bytes(value) for value in parameter_values)
        codebook_count = sum(int(value.numel()) for value in self.codebooks)
        bias_count = 0 if self.lm_head_bias is None else self.vocab_size
        assignment_bytes = _tensor_bytes(self.assignments)
        medoid_bytes = sum(_tensor_bytes(value) for value in self.medoid_source_indices)
        integer_buffer_bytes = sum(
            _tensor_bytes(value)
            for _, value in self.named_buffers()
            if value.dtype in _INTEGER_DTYPES
        )
        metadata_bytes = integer_buffer_bytes - assignment_bytes - medoid_bytes
        all_buffer_bytes = sum(
            _tensor_bytes(value) for _, value in self.named_buffers()
        )
        other_buffer_bytes = all_buffer_bytes - integer_buffer_bytes
        dense_weight_count = self.vocab_size * self.hidden_size
        dense_count = dense_weight_count + bias_count
        dense_bytes = dense_weight_count * int(self.codebooks[0].element_size())
        if self.lm_head_bias is not None:
            dense_bytes += _tensor_bytes(self.lm_head_bias)
        return CompositionalVocabularyReport(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            subspace_count=self.subspace_count,
            subspace_widths=self.subspace_widths,
            codebook_sizes=tuple(int(value) for value in self.codebook_sizes.tolist()),
            codebook_parameters=codebook_count,
            bias_parameters=bias_count,
            learned_parameter_count=learned_count,
            learned_parameter_bytes=learned_bytes,
            assignment_index_bytes=assignment_bytes,
            medoid_source_index_bytes=medoid_bytes,
            metadata_index_bytes=metadata_bytes,
            index_buffer_bytes=integer_buffer_bytes,
            other_tensor_buffer_bytes=other_buffer_bytes,
            tensor_payload_bytes=learned_bytes + all_buffer_bytes,
            dense_weight_parameter_count=dense_weight_count,
            dense_equivalent_parameter_count=dense_count,
            dense_equivalent_bytes=dense_bytes,
            actual_serialized_bytes=serialized_tensor_state_bytes(state_dict),
        )

    @property
    def report(self) -> CompositionalVocabularyReport:
        return self._report_for_state(self.state_dict())

    @property
    def diagnostics(self) -> CompositionalVocabularyFitDiagnostics:
        metadata = self._fit_metadata.detach().to(device="cpu", dtype=torch.int64)
        initialization_code = int(metadata[_META_INITIALIZATION].item())
        if initialization_code not in _INITIALIZATION_NAMES:
            raise CompositionalVocabularyStateError(
                f"Unknown fitting initialization code {initialization_code}."
            )
        codebook_sizes = tuple(int(value) for value in self.codebook_sizes.tolist())
        sample_sizes = tuple(
            int(value) for value in self._initialization_sample_sizes.tolist()
        )
        effective_token = tuple(
            int(value) for value in self._effective_token_chunk_sizes.tolist()
        )
        effective_codeword = tuple(
            int(value) for value in self._effective_codeword_chunk_sizes.tolist()
        )
        iterations = tuple(
            int(value) for value in self._iterations_per_subspace.tolist()
        )
        supplied = bool(int(metadata[_META_SUPPLIED_ASSIGNMENTS].item()))
        max_iterations = int(metadata[_META_MAX_ITERATIONS].item())
        if supplied:
            work_bound = self.vocab_size * self.subspace_count
        else:
            work_bound = sum(
                sample_size * codebook_size
                + max_iterations * self.vocab_size * (codebook_size + 1)
                for sample_size, codebook_size in zip(
                    sample_sizes, codebook_sizes, strict=True
                )
            )
        block_limit = max(
            token * codeword
            for token, codeword in zip(effective_token, effective_codeword, strict=True)
        )
        distances = int(metadata[_META_DISTANCE_VALUES].item())
        max_block = int(metadata[_META_MAX_DISTANCE_BLOCK].item())
        return CompositionalVocabularyFitDiagnostics(
            algorithm="deterministic_chunked_product_medoids",
            initialization=_INITIALIZATION_NAMES[initialization_code],
            deterministic=True,
            supplied_assignments=supplied,
            assignments_refined=not supplied,
            real_medoid_codewords=True,
            tie_break="lowest codeword, then lowest source token index",
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            subspace_count=self.subspace_count,
            subspace_widths=self.subspace_widths,
            codebook_sizes=codebook_sizes,
            seed=int(metadata[_META_SEED].item()),
            requested_initialization_minibatch_size=int(
                metadata[_META_REQUESTED_MINIBATCH].item()
            ),
            initialization_sample_sizes=sample_sizes,
            requested_token_chunk_size=int(
                metadata[_META_REQUESTED_TOKEN_CHUNK].item()
            ),
            requested_codeword_chunk_size=int(
                metadata[_META_REQUESTED_CODEWORD_CHUNK].item()
            ),
            effective_token_chunk_sizes=effective_token,
            effective_codeword_chunk_sizes=effective_codeword,
            max_iterations=max_iterations,
            iterations_per_subspace=iterations,
            iterations_run=int(metadata[_META_ITERATIONS_RUN].item()),
            distance_values_computed=distances,
            distance_work_bound=work_bound,
            max_distance_block_elements=max_block,
            distance_block_element_limit=block_limit,
            full_vocabulary_by_codebook_matrix_elements=(
                self.vocab_size * sum(codebook_sizes)
            ),
            largest_vocabulary_by_codebook_matrix_elements=(
                self.vocab_size * max(codebook_sizes)
            ),
            materialized_full_vocabulary_by_codebook_matrix=False,
            bounded_work=distances <= work_bound and max_block <= block_limit,
            fitting_requires_grad=False,
            squared_reconstruction_error=float(self._fit_squared_error.item()),
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
        *,
        prefix: str = "",
    ) -> CompositionalVocabularyRegistry:
        """Reconstruct exactly from ordinary tensors, with no opaque extra state."""

        if not isinstance(state_dict, Mapping) or any(
            not isinstance(value, Tensor) for value in state_dict.values()
        ):
            raise CompositionalVocabularyStateError(
                "state_dict must contain tensor values only."
            )
        if not prefix and "registry.assignments" in state_dict:
            prefix = "registry."
        elif prefix and not prefix.endswith("."):
            prefix = f"{prefix}."
        local = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        required = (
            "assignments",
            "subspace_offsets",
            "codebook_sizes",
            "_format_version",
            "_padding_idx",
            "_fit_metadata",
            "_iterations_per_subspace",
            "_initialization_sample_sizes",
            "_effective_token_chunk_sizes",
            "_effective_codeword_chunk_sizes",
            "_fit_squared_error",
        )
        missing = [name for name in required if name not in local]
        if missing:
            raise CompositionalVocabularyStateError(
                f"state_dict is missing required tensors {missing!r}."
            )
        version = local["_format_version"]
        if (
            version.numel() != 1
            or int(version.detach().cpu().item()) != _FORMAT_VERSION
        ):
            raise CompositionalVocabularyStateError(
                "state_dict has an unsupported compositional vocabulary version."
            )
        sizes = local["codebook_sizes"].detach().to(device="cpu", dtype=torch.int64)
        if sizes.ndim != 1 or int(sizes.numel()) <= 0:
            raise CompositionalVocabularyStateError(
                "codebook_sizes must be a non-empty vector."
            )
        subspace_count = int(sizes.numel())
        codebooks: list[Tensor] = []
        medoids: list[Tensor] = []
        for index in range(subspace_count):
            codebook_name = f"codebooks.{index}"
            medoid_name = f"medoid_source_indices_{index}"
            if codebook_name not in local or medoid_name not in local:
                raise CompositionalVocabularyStateError(
                    f"state_dict is missing {codebook_name!r} or {medoid_name!r}."
                )
            codebooks.append(local[codebook_name])
            medoids.append(local[medoid_name])
        padding_value = int(local["_padding_idx"].detach().cpu().item())
        result = cls(
            codebooks,
            local["assignments"],
            medoids,
            subspace_offsets=local["subspace_offsets"],
            lm_head_bias=local.get("lm_head_bias"),
            padding_idx=None if padding_value < 0 else padding_value,
            fit_metadata=local["_fit_metadata"],
            iterations_per_subspace=local["_iterations_per_subspace"],
            initialization_sample_sizes=local["_initialization_sample_sizes"],
            effective_token_chunk_sizes=local["_effective_token_chunk_sizes"],
            effective_codeword_chunk_sizes=local["_effective_codeword_chunk_sizes"],
            fit_squared_error=local["_fit_squared_error"],
        )
        incompatible = result.load_state_dict(local, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise CompositionalVocabularyStateError(
                "state_dict did not exactly match the reconstructed registry."
            )
        return result


class CompositionalEmbedding(nn.Module):
    """Parameterless embedding view backed by one product-codebook registry."""

    def __init__(self, registry: CompositionalVocabularyRegistry) -> None:
        super().__init__()
        if not isinstance(registry, CompositionalVocabularyRegistry):
            raise CompositionalVocabularyShapeError(
                "registry must be a CompositionalVocabularyRegistry."
            )
        object.__setattr__(self, "_registry", registry)
        self.num_embeddings = registry.vocab_size
        self.embedding_dim = registry.hidden_size
        self.padding_idx = registry.padding_idx

    @property
    def registry(self) -> CompositionalVocabularyRegistry:
        return self._registry

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.registry.embed(input_ids)

    def materialize_weight(self) -> Tensor:
        return self.registry.materialize_weight()

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"subspaces={self.registry.subspace_count}"
        )


class CompositionalLMHead(nn.Module):
    """Parameterless tied LM-head view using codeword scores and index gathers."""

    def __init__(self, registry: CompositionalVocabularyRegistry) -> None:
        super().__init__()
        if not isinstance(registry, CompositionalVocabularyRegistry):
            raise CompositionalVocabularyShapeError(
                "registry must be a CompositionalVocabularyRegistry."
            )
        object.__setattr__(self, "_registry", registry)
        self.in_features = registry.hidden_size
        self.out_features = registry.vocab_size

    @property
    def registry(self) -> CompositionalVocabularyRegistry:
        return self._registry

    @property
    def bias(self) -> nn.Parameter | None:
        return self.registry.lm_head_bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.registry.logits(hidden_states)

    def materialize_weight(self) -> Tensor:
        return self.registry.materialize_weight()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"subspaces={self.registry.subspace_count}, bias={self.bias is not None}"
        )


class TiedCompositionalVocabulary(nn.Module):
    """One registry plus tied, parameterless embedding and LM-head runtime views."""

    def __init__(self, registry: CompositionalVocabularyRegistry) -> None:
        super().__init__()
        if not isinstance(registry, CompositionalVocabularyRegistry):
            raise CompositionalVocabularyShapeError(
                "registry must be a CompositionalVocabularyRegistry."
            )
        self.registry = registry
        self.embedding = CompositionalEmbedding(registry)
        self.lm_head = CompositionalLMHead(registry)

    def __iter__(self) -> Iterator[nn.Module]:
        yield self.embedding
        yield self.lm_head

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)

    @property
    def report(self) -> CompositionalVocabularyReport:
        return self.registry._report_for_state(self.state_dict())

    @property
    def diagnostics(self) -> CompositionalVocabularyFitDiagnostics:
        return self.registry.diagnostics

    def materialize_weight(self) -> Tensor:
        return self.registry.materialize_weight()

    reconstruct_weight = materialize_weight

    def materialize(self) -> tuple[nn.Embedding, nn.Linear]:
        weight = self.materialize_weight()
        embedding = nn.Embedding(
            self.registry.vocab_size,
            self.registry.hidden_size,
            padding_idx=self.registry.padding_idx,
            device=weight.device,
            dtype=weight.dtype,
        )
        embedding.weight = nn.Parameter(weight, requires_grad=True)
        lm_head = nn.Linear(
            self.registry.hidden_size,
            self.registry.vocab_size,
            bias=self.registry.lm_head_bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        lm_head.weight = embedding.weight
        if self.registry.lm_head_bias is not None:
            lm_head.bias = nn.Parameter(
                self.registry.lm_head_bias.detach().clone(), requires_grad=True
            )
        embedding.train(self.training)
        lm_head.train(self.training)
        return embedding, lm_head

    to_dense = materialize

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
    ) -> TiedCompositionalVocabulary:
        prefixed = "registry.assignments" in state_dict
        registry = CompositionalVocabularyRegistry.from_state_dict(
            state_dict, prefix="registry" if prefixed else ""
        )
        result = cls(registry)
        if prefixed:
            incompatible = result.load_state_dict(state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise CompositionalVocabularyStateError(
                    "state_dict did not exactly match the reconstructed vocabulary."
                )
        return result


def _effective_chunks(
    vocab_size: int,
    codebook_size: int,
    token_chunk_size: int,
    codeword_chunk_size: int,
) -> tuple[int, int]:
    token_chunk = min(vocab_size, token_chunk_size)
    codeword_chunk = min(codebook_size, codeword_chunk_size)
    # For non-degenerate shapes, force at least one split even when callers set
    # both requested chunks above the whole problem dimensions.
    if vocab_size > 1 and token_chunk == vocab_size and codeword_chunk == codebook_size:
        token_chunk = max(1, vocab_size // 2)
    return token_chunk, codeword_chunk


def _squared_distances(points: Tensor, centers: Tensor) -> Tensor:
    point_norms = points.square().sum(dim=1, keepdim=True)
    center_norms = centers.square().sum(dim=1).unsqueeze(0)
    return (point_norms + center_norms - 2.0 * points.matmul(centers.t())).clamp_min_(
        0.0
    )


def _chunked_assign(
    points: Tensor,
    codebook: Tensor,
    *,
    token_chunk_size: int,
    codeword_chunk_size: int,
    tracker: _WorkTracker,
) -> tuple[Tensor, Tensor]:
    vocab_size = int(points.shape[0])
    codebook_size = int(codebook.shape[0])
    assignments = torch.empty(vocab_size, dtype=torch.int64)
    best_distances = torch.empty(vocab_size, dtype=torch.float64)
    for token_start in range(0, vocab_size, token_chunk_size):
        token_end = min(vocab_size, token_start + token_chunk_size)
        point_block = points[token_start:token_end]
        block_best = torch.full(
            (token_end - token_start,), math.inf, dtype=torch.float64
        )
        block_assignments = torch.zeros(token_end - token_start, dtype=torch.int64)
        for codeword_start in range(0, codebook_size, codeword_chunk_size):
            codeword_end = min(codebook_size, codeword_start + codeword_chunk_size)
            distances = _squared_distances(
                point_block, codebook[codeword_start:codeword_end]
            )
            tracker.add(int(distances.shape[0]), int(distances.shape[1]))
            values, local_indices = distances.min(dim=1)
            # Strict comparison preserves the lower codeword index on a tie.
            improve = values < block_best
            block_best = torch.where(improve, values, block_best)
            block_assignments = torch.where(
                improve, local_indices + codeword_start, block_assignments
            )
        assignments[token_start:token_end] = block_assignments
        best_distances[token_start:token_end] = block_best
    return assignments, best_distances


def _coprime_multiplier(length: int, mixed_seed: int) -> int:
    if length == 1:
        return 0
    candidate = (2 * mixed_seed + 1) % length
    if candidate == 0:
        candidate = 1
    while math.gcd(candidate, length) != 1:
        candidate = (candidate + 1) % length
        if candidate == 0:
            candidate = 1
    return candidate


def _hash_minibatch_indices(
    vocab_size: int,
    sample_size: int,
    *,
    seed: int,
    subspace_index: int,
) -> Tensor:
    mask = (1 << 63) - 1
    mixed = (seed + 0x4F1BBCDCBFA54001 * (subspace_index + 1)) & mask
    multiplier = _coprime_multiplier(vocab_size, mixed)
    offset = ((mixed >> 17) ^ mixed) % vocab_size
    order = torch.arange(sample_size, dtype=torch.int64)
    return (order * multiplier + offset).remainder(vocab_size)


def _farthest_first_real_rows(
    points: Tensor,
    sample_indices: Tensor,
    codebook_size: int,
    *,
    token_chunk_size: int,
    tracker: _WorkTracker,
) -> tuple[Tensor, Tensor]:
    sample = points.index_select(0, sample_indices)
    sample_size = int(sample.shape[0])
    selected_positions: list[int] = [0]
    selected_mask = torch.zeros(sample_size, dtype=torch.bool)
    nearest = torch.full((sample_size,), math.inf, dtype=torch.float64)
    while len(selected_positions) <= codebook_size:
        current_position = selected_positions[-1]
        selected_mask[current_position] = True
        center = sample[current_position : current_position + 1]
        for start in range(0, sample_size, token_chunk_size):
            end = min(sample_size, start + token_chunk_size)
            distances = _squared_distances(sample[start:end], center).squeeze(1)
            tracker.add(end - start, 1)
            nearest[start:end] = torch.minimum(nearest[start:end], distances)
        if len(selected_positions) == codebook_size:
            break
        candidates = nearest.clone()
        candidates[selected_mask] = -1.0
        maximum = candidates.max()
        tied_positions = torch.nonzero(candidates == maximum, as_tuple=False).flatten()
        tied_source_indices = sample_indices.index_select(0, tied_positions)
        next_in_ties = int(torch.argmin(tied_source_indices).item())
        selected_positions.append(int(tied_positions[next_in_ties].item()))
    positions = torch.tensor(selected_positions, dtype=torch.int64)
    source_indices = sample_indices.index_select(0, positions)
    return source_indices, points.index_select(0, source_indices)


def _exact_squared_euclidean_medoids(
    points: Tensor,
    assignments: Tensor,
    codebook_size: int,
    *,
    token_chunk_size: int,
    tracker: _WorkTracker,
) -> tuple[Tensor, Tensor]:
    vocab_size, width = map(int, points.shape)
    counts = torch.bincount(assignments, minlength=codebook_size)
    if bool((counts == 0).any()):
        raise CompositionalVocabularyShapeError(
            "every codeword must have at least one assigned token."
        )
    centroids = torch.zeros(codebook_size, width, dtype=torch.float64)
    centroids.index_add_(0, assignments, points)
    centroids /= counts.to(torch.float64).unsqueeze(1)

    distances = torch.empty(vocab_size, dtype=torch.float64)
    for start in range(0, vocab_size, token_chunk_size):
        end = min(vocab_size, start + token_chunk_size)
        labels = assignments[start:end]
        delta = points[start:end] - centroids.index_select(0, labels)
        distances[start:end] = delta.square().sum(dim=1)
        tracker.add(end - start, 1)

    minima = torch.full((codebook_size,), math.inf, dtype=torch.float64)
    minima.scatter_reduce_(0, assignments, distances, reduce="amin", include_self=True)
    token_indices = torch.arange(vocab_size, dtype=torch.int64)
    candidates = torch.where(
        distances == minima.index_select(0, assignments),
        token_indices,
        torch.full_like(token_indices, vocab_size),
    )
    medoid_indices = torch.full((codebook_size,), vocab_size, dtype=torch.int64)
    medoid_indices.scatter_reduce_(
        0, assignments, candidates, reduce="amin", include_self=True
    )
    if bool((medoid_indices == vocab_size).any()):  # pragma: no cover - guarded above
        raise RuntimeError("Failed to select one medoid for every codeword.")
    return medoid_indices, points.index_select(0, medoid_indices)


def _fit_one_subspace(
    points: Tensor,
    codebook_size: int,
    *,
    seed: int,
    subspace_index: int,
    max_iterations: int,
    token_chunk_size: int,
    codeword_chunk_size: int,
    initialization_minibatch_size: int,
    tracker: _WorkTracker,
) -> tuple[Tensor, Tensor, Tensor, int, int]:
    vocab_size = int(points.shape[0])
    sample_size = min(vocab_size, max(codebook_size, initialization_minibatch_size))
    sample_indices = _hash_minibatch_indices(
        vocab_size,
        sample_size,
        seed=seed,
        subspace_index=subspace_index,
    )
    medoid_indices, codebook = _farthest_first_real_rows(
        points,
        sample_indices,
        codebook_size,
        token_chunk_size=token_chunk_size,
        tracker=tracker,
    )
    previous_assignments: Tensor | None = None
    iterations_run = 0
    for _ in range(max_iterations):
        assignments, _ = _chunked_assign(
            points,
            codebook,
            token_chunk_size=token_chunk_size,
            codeword_chunk_size=codeword_chunk_size,
            tracker=tracker,
        )
        # Keep each real center in its own cluster, preventing empty codewords
        # without introducing synthetic centroids or unbounded reseeding work.
        assignments[medoid_indices] = torch.arange(codebook_size, dtype=torch.int64)
        new_indices, new_codebook = _exact_squared_euclidean_medoids(
            points,
            assignments,
            codebook_size,
            token_chunk_size=token_chunk_size,
            tracker=tracker,
        )
        iterations_run += 1
        converged = (
            previous_assignments is not None
            and torch.equal(assignments, previous_assignments)
            and torch.equal(new_indices, medoid_indices)
        )
        previous_assignments = assignments
        medoid_indices = new_indices
        codebook = new_codebook
        if converged:
            break
    if previous_assignments is None:  # pragma: no cover - max_iterations is positive
        raise RuntimeError("Product-medoid fitting did not run an iteration.")
    return (
        previous_assignments,
        codebook,
        medoid_indices,
        iterations_run,
        sample_size,
    )


def _resolve_subspace_count(
    *,
    num_subspaces: int | None,
    subspace_count: int | None,
    assignments: Tensor | None,
    codebook_size: int | Sequence[int] | None,
) -> int:
    if num_subspaces is not None and subspace_count is not None:
        if num_subspaces != subspace_count:
            raise CompositionalVocabularyShapeError(
                "num_subspaces and subspace_count disagree."
            )
    selected = num_subspaces if num_subspaces is not None else subspace_count
    if selected is None and assignments is not None and assignments.ndim == 2:
        selected = int(assignments.shape[1])
    if (
        selected is None
        and isinstance(codebook_size, Sequence)
        and not isinstance(codebook_size, (str, bytes))
    ):
        selected = len(codebook_size)
    if selected is None:
        raise CompositionalVocabularyShapeError(
            "num_subspaces (or subspace_count) must be supplied."
        )
    return _positive_int(selected, "num_subspaces")


def _resolve_codebook_sizes(
    value: int | Sequence[int] | None,
    *,
    subspace_count: int,
    vocab_size: int,
    assignments: Tensor | None,
) -> tuple[int, ...]:
    if value is None:
        if assignments is None:
            raise CompositionalVocabularyShapeError(
                "codebook_size must be supplied when assignments are not supplied."
            )
        assignment64 = assignments.detach().to(device="cpu", dtype=torch.int64)
        sizes = tuple(
            int(assignment64[:, index].max().item()) + 1
            for index in range(subspace_count)
        )
    elif isinstance(value, int) and not isinstance(value, bool):
        size = _positive_int(value, "codebook_size")
        sizes = (size,) * subspace_count
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sizes = tuple(
            _positive_int(item, f"codebook_size[{index}]")
            for index, item in enumerate(value)
        )
        if len(sizes) != subspace_count:
            raise CompositionalVocabularyShapeError(
                "codebook_size sequence length must equal num_subspaces."
            )
    else:
        raise CompositionalVocabularyShapeError(
            "codebook_size must be an integer or integer sequence."
        )
    if any(size > vocab_size for size in sizes):
        raise CompositionalVocabularyShapeError(
            "a real-medoid codebook cannot contain more rows than the vocabulary."
        )
    return sizes


def _source_weight_and_bias(
    source: Tensor | nn.Embedding,
    lm_head: nn.Linear | None,
    explicit_bias: Tensor | None,
) -> tuple[Tensor, Tensor | None, int | None, bool]:
    if isinstance(source, nn.Embedding):
        weight = _validate_float_tensor(source.weight, "embedding.weight", 2)
        padding_idx = source.padding_idx
        training = source.training
        if lm_head is not None:
            if not isinstance(lm_head, nn.Linear):
                raise CompositionalVocabularyShapeError(
                    "lm_head must be torch.nn.Linear."
                )
            head_weight = _validate_float_tensor(lm_head.weight, "lm_head.weight", 2)
            if tuple(head_weight.shape) != tuple(weight.shape):
                raise CompositionalVocabularyShapeError(
                    "embedding and LM head must have matching weight shapes."
                )
            if not _same_tensor_storage(weight, head_weight):
                raise CompositionalVocabularyShapeError(
                    "embedding and LM head weights must be truly tied."
                )
            if explicit_bias is not None:
                raise CompositionalVocabularyShapeError(
                    "bias must be omitted when lm_head supplies its bias."
                )
            bias = lm_head.bias
            training = source.training or lm_head.training
        else:
            bias = explicit_bias
    elif isinstance(source, Tensor):
        if lm_head is not None:
            raise CompositionalVocabularyShapeError(
                "lm_head can only accompany an nn.Embedding source."
            )
        weight = _validate_float_tensor(source, "source_weight", 2)
        padding_idx = None
        training = True
        bias = explicit_bias
    else:
        raise CompositionalVocabularyShapeError(
            "source must be a weight tensor or torch.nn.Embedding."
        )
    if bias is not None:
        bias = _validate_float_tensor(bias, "bias", 1)
        if int(bias.numel()) != int(weight.shape[0]):
            raise CompositionalVocabularyShapeError(
                "bias must contain one value per vocabulary token."
            )
        if bias.device != weight.device or bias.dtype != weight.dtype:
            raise CompositionalVocabularyShapeError(
                "bias must match source weight dtype and device."
            )
    return weight, bias, padding_idx, training


def _validated_supplied_assignments(
    assignments: Tensor,
    *,
    vocab_size: int,
    subspace_count: int,
    codebook_sizes: tuple[int, ...],
) -> Tensor:
    assignments = _validate_integer_matrix(assignments, "assignments")
    if tuple(assignments.shape) != (vocab_size, subspace_count):
        raise CompositionalVocabularyShapeError(
            f"assignments must have shape ({vocab_size}, {subspace_count})."
        )
    result = assignments.detach().to(device="cpu", dtype=torch.int64).clone()
    for index, codebook_size in enumerate(codebook_sizes):
        column = result[:, index]
        if int(column.min()) < 0 or int(column.max()) >= codebook_size:
            raise CompositionalVocabularyShapeError(
                f"assignments column {index} has an out-of-range codeword."
            )
        if int(torch.unique(column).numel()) != codebook_size:
            raise CompositionalVocabularyShapeError(
                "supplied assignments must use every codeword in every subspace."
            )
    return result


def _chunked_reconstruction_error(
    source: Tensor,
    subspace_ranges: tuple[tuple[int, int], ...],
    codebooks: Sequence[Tensor],
    assignments: Tensor,
    *,
    token_chunk_size: int,
) -> float:
    total = 0.0
    vocab_size = int(source.shape[0])
    for index, ((start, end), codebook) in enumerate(
        zip(subspace_ranges, codebooks, strict=True)
    ):
        for token_start in range(0, vocab_size, token_chunk_size):
            token_end = min(vocab_size, token_start + token_chunk_size)
            labels = assignments[token_start:token_end, index]
            reconstructed = codebook.index_select(0, labels)
            delta = source[token_start:token_end, start:end] - reconstructed
            total += float(delta.square().sum().item())
    if not math.isfinite(total):
        raise CompositionalVocabularyFiniteError(
            "fitting produced a non-finite reconstruction error."
        )
    return total


def fit_tied_compositional_vocabulary(
    source: Tensor | nn.Embedding,
    lm_head: nn.Linear | None = None,
    *,
    num_subspaces: int | None = None,
    subspace_count: int | None = None,
    codebook_size: int | Sequence[int] | None = None,
    codewords_per_subspace: int | Sequence[int] | None = None,
    assignments: Tensor | None = None,
    seed: int = 0,
    max_iterations: int = 4,
    token_chunk_size: int = 1024,
    codeword_chunk_size: int = 128,
    initialization_minibatch_size: int = 4096,
    minibatch_size: int | None = None,
    bias: Tensor | None = None,
) -> TiedCompositionalVocabulary:
    """Fit deterministic real-row product codebooks for one tied vocabulary.

    Supplied assignments are kept fixed and receive exact squared-Euclidean
    medoids.  Otherwise a deterministic affine-hash minibatch chooses bounded
    farthest-first real-row seeds, followed by a bounded number of chunked
    assignment/medoid updates.  No fitting path creates a full ``V x K``
    distance matrix.
    """

    if codebook_size is not None and codewords_per_subspace is not None:
        if codebook_size != codewords_per_subspace:
            raise CompositionalVocabularyShapeError(
                "codebook_size and codewords_per_subspace disagree."
            )
    selected_codebook_size = (
        codebook_size if codebook_size is not None else codewords_per_subspace
    )
    if minibatch_size is not None:
        if initialization_minibatch_size != 4096:
            raise CompositionalVocabularyShapeError(
                "Specify only one minibatch size keyword."
            )
        initialization_minibatch_size = minibatch_size
    seed = _nonnegative_int(seed, "seed")
    if seed > torch.iinfo(torch.int64).max:
        raise CompositionalVocabularyShapeError("seed exceeds signed int64 range.")
    max_iterations = _positive_int(max_iterations, "max_iterations")
    token_chunk_size = _positive_int(token_chunk_size, "token_chunk_size")
    codeword_chunk_size = _positive_int(codeword_chunk_size, "codeword_chunk_size")
    initialization_minibatch_size = _positive_int(
        initialization_minibatch_size, "initialization_minibatch_size"
    )
    weight, head_bias, padding_idx, training = _source_weight_and_bias(
        source, lm_head, bias
    )
    vocab_size, hidden_size = map(int, weight.shape)
    selected_subspaces = _resolve_subspace_count(
        num_subspaces=num_subspaces,
        subspace_count=subspace_count,
        assignments=assignments,
        codebook_size=selected_codebook_size,
    )
    subspace_ranges = split_hidden_subspaces(hidden_size, selected_subspaces)
    codebook_sizes = _resolve_codebook_sizes(
        selected_codebook_size,
        subspace_count=selected_subspaces,
        vocab_size=vocab_size,
        assignments=assignments,
    )
    supplied = None
    if assignments is not None:
        supplied = _validated_supplied_assignments(
            assignments,
            vocab_size=vocab_size,
            subspace_count=selected_subspaces,
            codebook_sizes=codebook_sizes,
        )

    source_cpu = weight.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(source_cpu).all()):
        raise CompositionalVocabularyFiniteError(
            "Casting source_weight to float64 produced NaN or infinity."
        )
    tracker = _WorkTracker()
    fitted_assignments: list[Tensor] = []
    fitted_codebooks_cpu: list[Tensor] = []
    medoid_indices: list[Tensor] = []
    iteration_counts: list[int] = []
    sample_sizes: list[int] = []
    effective_token_chunks: list[int] = []
    effective_codeword_chunks: list[int] = []

    for index, ((start, end), current_codebook_size) in enumerate(
        zip(subspace_ranges, codebook_sizes, strict=True)
    ):
        token_chunk, codeword_chunk = _effective_chunks(
            vocab_size,
            current_codebook_size,
            token_chunk_size,
            codeword_chunk_size,
        )
        effective_token_chunks.append(token_chunk)
        effective_codeword_chunks.append(codeword_chunk)
        points = source_cpu[:, start:end]
        if supplied is not None:
            current_assignments = supplied[:, index]
            current_medoids, current_codebook = _exact_squared_euclidean_medoids(
                points,
                current_assignments,
                current_codebook_size,
                token_chunk_size=token_chunk,
                tracker=tracker,
            )
            iterations_run = 0
            sample_size = 0
        else:
            (
                current_assignments,
                current_codebook,
                current_medoids,
                iterations_run,
                sample_size,
            ) = _fit_one_subspace(
                points,
                current_codebook_size,
                seed=seed,
                subspace_index=index,
                max_iterations=max_iterations,
                token_chunk_size=token_chunk,
                codeword_chunk_size=codeword_chunk,
                initialization_minibatch_size=initialization_minibatch_size,
                tracker=tracker,
            )
        fitted_assignments.append(current_assignments)
        fitted_codebooks_cpu.append(current_codebook)
        medoid_indices.append(current_medoids)
        iteration_counts.append(iterations_run)
        sample_sizes.append(sample_size)

    assignment_matrix = torch.stack(fitted_assignments, dim=1)
    squared_error = _chunked_reconstruction_error(
        source_cpu,
        subspace_ranges,
        fitted_codebooks_cpu,
        assignment_matrix,
        token_chunk_size=min(effective_token_chunks),
    )
    fit_metadata = torch.tensor(
        [
            _INITIALIZATION_CODES[
                "supplied_assignments"
                if supplied is not None
                else "hash_minibatch_farthest_first"
            ],
            seed,
            max_iterations,
            sum(iteration_counts),
            token_chunk_size,
            codeword_chunk_size,
            initialization_minibatch_size,
            tracker.distance_values,
            tracker.max_block_elements,
            int(supplied is not None),
        ],
        dtype=torch.int64,
    )
    offsets = torch.tensor(
        [subspace_ranges[0][0], *(end for _, end in subspace_ranges)],
        dtype=torch.int64,
    )
    codebooks = tuple(
        value.to(device=weight.device, dtype=weight.dtype).contiguous()
        for value in fitted_codebooks_cpu
    )
    registry = CompositionalVocabularyRegistry(
        codebooks,
        assignment_matrix.to(device=weight.device, dtype=torch.int32),
        tuple(
            value.to(device=weight.device, dtype=torch.int32)
            for value in medoid_indices
        ),
        subspace_offsets=offsets,
        lm_head_bias=head_bias,
        padding_idx=padding_idx,
        fit_metadata=fit_metadata,
        iterations_per_subspace=torch.tensor(iteration_counts, dtype=torch.int64),
        initialization_sample_sizes=torch.tensor(sample_sizes, dtype=torch.int64),
        effective_token_chunk_sizes=torch.tensor(
            effective_token_chunks, dtype=torch.int64
        ),
        effective_codeword_chunk_sizes=torch.tensor(
            effective_codeword_chunks, dtype=torch.int64
        ),
        fit_squared_error=torch.tensor(squared_error, dtype=torch.float64),
    )
    result = TiedCompositionalVocabulary(registry)
    result.train(training)
    return result


def materialize_compositional_vocabulary(
    vocabulary: TiedCompositionalVocabulary | CompositionalVocabularyRegistry,
) -> Tensor:
    """Explicitly reconstruct the represented dense tied vocabulary weight."""

    if isinstance(vocabulary, TiedCompositionalVocabulary):
        return vocabulary.materialize_weight()
    if isinstance(vocabulary, CompositionalVocabularyRegistry):
        return vocabulary.materialize_weight()
    raise CompositionalVocabularyShapeError(
        "vocabulary must be a tied compositional vocabulary or registry."
    )


ProductCodebookVocabularyRegistry = CompositionalVocabularyRegistry
ProductCodebookEmbedding = CompositionalEmbedding
ProductCodebookLMHead = CompositionalLMHead
TiedProductCodebookVocabulary = TiedCompositionalVocabulary
fit_compositional_vocabulary = fit_tied_compositional_vocabulary
fit_product_codebook_vocabulary = fit_tied_compositional_vocabulary
build_tied_compositional_vocabulary = fit_tied_compositional_vocabulary


__all__ = [
    "CompositionalEmbedding",
    "CompositionalLMHead",
    "CompositionalVocabularyError",
    "CompositionalVocabularyFiniteError",
    "CompositionalVocabularyFitDiagnostics",
    "CompositionalVocabularyRegistry",
    "CompositionalVocabularyReport",
    "CompositionalVocabularyShapeError",
    "CompositionalVocabularyStateError",
    "ProductCodebookEmbedding",
    "ProductCodebookLMHead",
    "ProductCodebookVocabularyRegistry",
    "TiedCompositionalVocabulary",
    "TiedProductCodebookVocabulary",
    "build_tied_compositional_vocabulary",
    "fit_compositional_vocabulary",
    "fit_product_codebook_vocabulary",
    "fit_tied_compositional_vocabulary",
    "materialize_compositional_vocabulary",
    "serialized_tensor_state_bytes",
    "split_hidden_subspaces",
]
