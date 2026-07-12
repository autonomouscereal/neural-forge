"""Memory-bounded two-pass sketches of an output Jacobian update.

For row batches ``D`` and ``X`` this module sketches, but never forms, the
matrix ``G = sum(D.T @ X)``.  An inherited low-rank product ``L @ R`` defines
occupied output and input spaces.  Both passes remove those spaces before
finding right directions which are useful for a new residual factor.

The accumulator deliberately owns no model hooks.  A caller may replay any
source of finite ``(D, X)`` batches through the two explicit passes.  Matrix
products and accumulators are FP32.  Rank-revealing decompositions are small,
CPU FP64 operations.  Every allocation made by the accumulator is covered by
the declared byte ceiling; caller-owned input batches are outside that
ceiling.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor


OUTPUT_JACOBIAN_SKETCH_SCHEMA = "neural-forge-output-jacobian-sketch-v2"
MINIMUM_NORMAL_FRACTION = 0.20
MINIMUM_GUIDED_OVER_RANDOM_RATIO = 1.50


class OutputJacobianSketchError(RuntimeError):
    """Base failure for an uncertifiable output-Jacobian sketch."""


class OutputJacobianSketchShapeError(OutputJacobianSketchError):
    """A factor, batch, or option has an invalid shape or value."""


class OutputJacobianSketchRankError(OutputJacobianSketchShapeError):
    """A requested or selected singular subspace is uncertifiable."""


class OutputJacobianSketchFiniteError(OutputJacobianSketchError):
    """An input or derived tensor is non-finite."""


class OutputJacobianSketchPhaseError(OutputJacobianSketchError):
    """A pass transition or replay ordering is invalid."""


class OutputJacobianSketchLimitError(OutputJacobianSketchError):
    """The fail-closed owned-memory ceiling would be exceeded."""


SketchPhase = Literal["pass1", "pass2", "finalized"]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise OutputJacobianSketchShapeError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise OutputJacobianSketchShapeError(f"{name} must be positive.") from error
    if result <= 0:
        raise OutputJacobianSketchShapeError(f"{name} must be positive.")
    return result


def _seed(value: Any) -> int:
    if isinstance(value, bool):
        raise OutputJacobianSketchShapeError("seed must be an integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise OutputJacobianSketchShapeError("seed must be an integer.") from error
    if result < -(2**63) or result >= 2**63:
        raise OutputJacobianSketchShapeError("seed must fit a signed 64-bit integer.")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise OutputJacobianSketchShapeError(f"{name} must be nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise OutputJacobianSketchShapeError(
            f"{name} must be nonnegative."
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise OutputJacobianSketchShapeError(f"{name} must be nonnegative.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _nonnegative_finite(value, name)
    if result == 0.0:
        raise OutputJacobianSketchShapeError(f"{name} must be positive.")
    return result


def _derived_seed(seed: int, label: str, attempt: int = 0) -> int:
    payload = f"{OUTPUT_JACOBIAN_SKETCH_SCHEMA}:{seed}:{label}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    """Hash tensor metadata and bytes without retaining a second full tensor."""

    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    # Flatten elements before reinterpreting their bytes.  Reinterpreting a
    # multidimensional tensor first changes only its last dimension for
    # multi-byte dtypes and is not a canonical flat payload contract.
    flat = tensor.reshape(-1).view(torch.uint8)
    # ``memoryview`` exposes the contiguous Torch CPU buffer through its NumPy
    # bridge without a payload-sized ``tobytes`` allocation.
    chunk_bytes = 1 << 20
    for start in range(0, int(flat.numel()), chunk_bytes):
        chunk = flat[start : start + chunk_bytes]
        digest.update(memoryview(chunk.numpy()))
    return digest.hexdigest()


def _tensor_bytes(value: Tensor) -> int:
    return int(value.numel() * value.element_size())


def _validate_dense_float_metadata(value: Any, name: str) -> Tensor:
    """Validate allocation-free tensor metadata before reading its values."""

    if not isinstance(value, Tensor):
        raise OutputJacobianSketchShapeError(f"{name} must be a torch tensor.")
    if value.device.type == "meta":
        raise OutputJacobianSketchShapeError(f"{name} cannot be a meta tensor.")
    if value.layout != torch.strided:
        raise OutputJacobianSketchShapeError(
            f"{name} must use dense strided layout."
        )
    supported_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if value.dtype not in supported_dtypes:
        raise OutputJacobianSketchShapeError(
            f"{name} must have a real floating-point dtype."
        )
    if value.is_conj() or value.is_neg():
        raise OutputJacobianSketchShapeError(
            f"{name} cannot carry conjugate or negative view metadata."
        )
    if value.device.type == "cuda":
        if not torch.cuda.is_available():
            raise OutputJacobianSketchShapeError(
                f"{name} is CUDA-resident but CUDA is unavailable."
            )
        index = value.device.index
        if index is None or index < 0 or index >= torch.cuda.device_count():
            raise OutputJacobianSketchShapeError(
                f"{name} has an invalid CUDA device index."
            )
    return value


def _canonicalize_columns(value: Tensor) -> Tensor:
    result = value.clone()
    for column in range(int(result.shape[1])):
        direction = result[:, column]
        pivot = int(torch.argmax(direction.abs()).item())
        if float(direction[pivot].item()) < 0.0:
            result[:, column].neg_()
    return result


def _canonicalize_rows(value: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    result = value.clone()
    pivots: list[int] = []
    for row in range(int(result.shape[0])):
        direction = result[row]
        pivot = int(torch.argmax(direction.abs()).item())
        pivots.append(pivot)
        if float(direction[pivot].item()) < 0.0:
            result[row].neg_()
    return result, tuple(pivots)


def _coordinate_canonical_subspace(value: Tensor) -> Tensor:
    """Choose a basis invariant to rotations inside an exact subspace."""

    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise OutputJacobianSketchShapeError(
            "Coordinate canonicalization requires a CPU FP64 basis."
        )
    width = int(value.shape[1])
    coefficient_basis = value.new_empty((width, 0))
    directions: list[Tensor] = []
    tolerance = (
        64.0
        * max(value.shape)
        * torch.finfo(torch.float64).eps
    )
    for coordinate in range(int(value.shape[0])):
        coefficient = value[coordinate].clone()
        if coefficient_basis.numel() > 0:
            coefficient = coefficient - coefficient_basis @ (
                coefficient_basis.T @ coefficient
            )
        norm = float(torch.linalg.vector_norm(coefficient).item())
        if norm <= tolerance:
            continue
        coefficient.div_(norm)
        direction = value @ coefficient
        if float(direction[coordinate].item()) < 0.0:
            coefficient.neg_()
            direction.neg_()
        coefficient_basis = torch.cat(
            (coefficient_basis, coefficient.reshape(-1, 1)), dim=1
        )
        directions.append(direction)
        if len(directions) == width:
            break
    if len(directions) != width:
        raise OutputJacobianSketchRankError(
            "A degenerate selected subspace has no canonical coordinate basis."
        )
    result = torch.stack(directions, dim=1)
    if not bool(torch.isfinite(result).all()):
        raise OutputJacobianSketchFiniteError(
            "Coordinate canonicalization became non-finite."
        )
    return result


def _rademacher(rows: int, columns: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(
        0,
        2,
        (rows, columns),
        dtype=torch.int8,
        device="cpu",
        generator=generator,
    )
    return values.to(torch.float32).mul_(2.0).sub_(1.0)


@dataclass(frozen=True, slots=True)
class OutputJacobianSketchSpec:
    """Dimensions, deterministic seed, gates, and hard owned-byte ceiling."""

    out_features: int
    in_features: int
    rank: int
    sketch_width: int
    seed: int = 0
    accumulator_device: str | None = None
    max_bytes: int = 512 * 1024 * 1024
    rank_rtol: float = 1e-6
    product_zero_rtol: float = 1e-7
    min_normal_fraction: float = MINIMUM_NORMAL_FRACTION
    min_guided_capture_fraction: float = 0.0
    min_guided_over_random_ratio: float = MINIMUM_GUIDED_OVER_RANDOM_RATIO

    def __post_init__(self) -> None:
        for name in (
            "out_features",
            "in_features",
            "rank",
            "sketch_width",
            "max_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(self, "seed", _seed(self.seed))
        if self.accumulator_device is not None:
            if not isinstance(self.accumulator_device, str):
                raise OutputJacobianSketchShapeError(
                    "accumulator_device must be a torch device string or None."
                )
            try:
                normalized_device = str(torch.device(self.accumulator_device))
            except (TypeError, ValueError, RuntimeError) as error:
                raise OutputJacobianSketchShapeError(
                    "accumulator_device is not a valid torch device."
                ) from error
            object.__setattr__(self, "accumulator_device", normalized_device)
        for name in ("rank_rtol", "product_zero_rtol"):
            object.__setattr__(
                self, name, _positive_finite(getattr(self, name), name)
            )
        for name in (
            "min_normal_fraction",
            "min_guided_capture_fraction",
            "min_guided_over_random_ratio",
        ):
            object.__setattr__(
                self, name, _nonnegative_finite(getattr(self, name), name)
            )
        if self.rank > min(self.out_features, self.in_features):
            raise OutputJacobianSketchShapeError(
                "rank cannot exceed either Jacobian dimension."
            )
        if self.rank > self.sketch_width:
            raise OutputJacobianSketchShapeError(
                "rank cannot exceed sketch_width."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return _json_sha256(
            {"schema": OUTPUT_JACOBIAN_SKETCH_SCHEMA, "spec": self.to_dict()}
        )


def _canonical_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise OutputJacobianSketchShapeError(
            f"{name} must be a lowercase SHA-256 string."
        )
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise OutputJacobianSketchShapeError(
            f"{name} must be a lowercase SHA-256 string."
        )
    return value


@dataclass(frozen=True, slots=True)
class OutputJacobianBlockIdentity:
    """Replay identity without requiring each site to hash full GPU tensors.

    ``semantic_identity`` binds stable model/prompt/prefix/site metadata.  The
    numeric fingerprint is a digest of the caller's declared numeric
    authenticator (or an exact tensor digest in the one-shot helper).  Generic
    streaming identities do not imply collision-free tensor authentication.
    Exact D/X hashes are optional for streaming hook integrations and are
    always populated by :func:`sketch_output_jacobian`.
    """

    semantic_identity: str
    numeric_fingerprint_sha256: str
    delta_sha256: str | None = None
    inputs_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_identity, str):
            raise OutputJacobianSketchShapeError(
                "semantic_identity must be a non-empty string."
            )
        try:
            encoded = self.semantic_identity.encode("utf-8")
        except UnicodeEncodeError as error:
            raise OutputJacobianSketchShapeError(
                "semantic_identity must be valid UTF-8 text."
            ) from error
        if not encoded or len(encoded) > 64 * 1024:
            raise OutputJacobianSketchShapeError(
                "semantic_identity must contain 1 to 65536 UTF-8 bytes."
            )
        object.__setattr__(
            self,
            "numeric_fingerprint_sha256",
            _canonical_sha256(
                self.numeric_fingerprint_sha256,
                "numeric_fingerprint_sha256",
            ),
        )
        if (self.delta_sha256 is None) != (self.inputs_sha256 is None):
            raise OutputJacobianSketchShapeError(
                "delta_sha256 and inputs_sha256 must be supplied together."
            )
        if self.delta_sha256 is not None:
            object.__setattr__(
                self,
                "delta_sha256",
                _canonical_sha256(self.delta_sha256, "delta_sha256"),
            )
            assert self.inputs_sha256 is not None
            object.__setattr__(
                self,
                "inputs_sha256",
                _canonical_sha256(self.inputs_sha256, "inputs_sha256"),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return _json_sha256(
            {
                "schema": f"{OUTPUT_JACOBIAN_SKETCH_SCHEMA}:block-identity-v1",
                **self.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class OutputJacobianSketchReport:
    """Auditable estimates, controls, gates, hashes, and memory accounting."""

    schema: str
    spec_sha256: str
    accumulator_device: str
    sample_count: int
    block_count: int
    probe_count: int
    range_basis_width: int
    normal_effective_rank: int
    selected_singular_values: tuple[float, ...]
    inherited_latent_rank: int
    occupied_output_rank: int
    occupied_input_rank: int
    inherited_product_frobenius: float
    inherited_product_numerically_zero: bool
    raw_frobenius_sq_estimate: float
    normal_frobenius_sq_estimate: float
    normal_fraction: float
    guided_capture_energy: float
    guided_capture_fraction: float
    random_capture_energy: float
    random_capture_fraction: float
    guided_over_random_ratio: float
    minimum_normal_fraction: float
    minimum_guided_capture_fraction: float
    minimum_guided_over_random_ratio: float
    normal_fraction_gate_passed: bool
    guided_capture_gate_passed: bool
    guided_over_random_gate_passed: bool
    gate_passed: bool
    accounted_peak_bytes: int
    max_bytes: int
    probe_sha256: str
    inherited_left_sha256: str
    inherited_right_sha256: str
    occupied_output_basis_sha256: str
    occupied_input_basis_sha256: str
    pass1_normal_sketch_sha256: str
    range_basis_sha256: str
    random_control_sha256: str
    pass2_guided_core_sha256: str
    pass2_random_core_sha256: str
    guided_right_rows_sha256: str
    random_right_rows_sha256: str
    pass1_block_identity_chain_sha256: str
    pass2_block_identity_chain_sha256: str
    block_identity_replay_authenticated: bool
    guided_sign_pivots: tuple[int, ...]
    random_sign_pivots: tuple[int, ...]
    report_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OutputJacobianSketchResult:
    """Canonical guided right rows and an equal-rank random control."""

    guided_right_rows: Tensor
    random_right_rows: Tensor
    report: OutputJacobianSketchReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "guided_right_rows_shape": list(self.guided_right_rows.shape),
            "random_right_rows_shape": list(self.random_right_rows.shape),
            "report": self.report.to_dict(),
        }


class TwoPassOutputJacobianSketch:
    """Streaming, deterministic two-pass accumulator for ``sum(D.T @ X)``."""

    def __init__(
        self,
        spec: OutputJacobianSketchSpec,
        inherited_left: Tensor | None = None,
        inherited_right: Tensor | None = None,
        *,
        accumulator_device: str | torch.device | None = None,
    ) -> None:
        if not isinstance(spec, OutputJacobianSketchSpec):
            raise OutputJacobianSketchShapeError(
                "spec must be an OutputJacobianSketchSpec."
            )
        self.spec = spec
        self._phase: SketchPhase = "pass1"
        self._peak_bytes = 0
        self._pass1_samples = 0
        self._pass1_blocks = 0
        self._pass2_samples = 0
        self._pass2_blocks = 0
        identity_header = (
            f"{OUTPUT_JACOBIAN_SKETCH_SCHEMA}:ordered-block-identities-v1"
        ).encode()
        self._pass1_identity_chain = hashlib.sha256(identity_header)
        self._pass2_identity_chain = hashlib.sha256(identity_header)
        self._pass1_identity_sha256 = ""
        self._pass2_identity_sha256 = ""
        self._pass1_prevalidated = False
        self._pass2_prevalidated = False

        left, right, latent_rank = self._validate_factors(
            inherited_left, inherited_right
        )
        self._latent_rank = latent_rank
        self._left_hash = _tensor_sha256(left)
        self._right_hash = _tensor_sha256(right)
        requested_device = (
            accumulator_device
            if accumulator_device is not None
            else spec.accumulator_device
        )
        if requested_device is None:
            requested_device = left.device if inherited_left is not None else "cpu"
        try:
            self._device = torch.device(requested_device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise OutputJacobianSketchShapeError(
                "accumulator_device is not a valid torch device."
            ) from error
        if self._device.type not in {"cpu", "cuda"}:
            raise OutputJacobianSketchShapeError(
                "accumulator_device must be CPU or CUDA."
            )
        if self._device.type == "cuda":
            if not torch.cuda.is_available():
                raise OutputJacobianSketchShapeError(
                    "A CUDA accumulator_device was requested but CUDA is unavailable."
                )
            device_index = self._device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise OutputJacobianSketchShapeError(
                    "accumulator_device has an invalid CUDA device index."
                )
            self._device = torch.device("cuda", device_index)
        self._u, self._v, product_frobenius, product_zero = self._occupied_spaces(
            left, right
        )
        self._product_frobenius = product_frobenius
        self._product_zero = product_zero

        normal_input_dimensions = spec.in_features - int(self._v.shape[1])
        if normal_input_dimensions < spec.rank:
            raise OutputJacobianSketchShapeError(
                "The unoccupied input space cannot support the requested rank."
            )
        self._range_width = min(spec.sketch_width, normal_input_dimensions)

        prospective = self._persistent_bytes_for(
            probe=True,
            pass1=True,
            range_basis=False,
            pass2=False,
        )
        omega_bytes = 4 * spec.out_features * spec.sketch_width
        self._require_bytes(
            prospective + omega_bytes,
            "initial sketch allocation and deterministic CPU probe staging",
            absolute=True,
        )
        self._omega = _rademacher(
            spec.out_features,
            spec.sketch_width,
            _derived_seed(spec.seed, "range-and-hutchinson-probes"),
        ).to(self._device)
        self._raw_y = torch.zeros(
            (spec.in_features, spec.sketch_width),
            dtype=torch.float32,
            device=self._device,
        )
        self._normal_y = torch.zeros_like(self._raw_y)
        self._q: Tensor | None = None
        self._control: Tensor | None = None
        self._b: Tensor | None = None
        self._random_b: Tensor | None = None
        self._raw_frobenius_sq = 0.0
        self._normal_frobenius_sq = 0.0
        self._probe_hash = _tensor_sha256(self._omega)
        self._pass1_normal_hash = ""

    @property
    def phase(self) -> SketchPhase:
        return self._phase

    @property
    def accumulator_device(self) -> torch.device:
        return self._device

    @property
    def accounted_peak_bytes(self) -> int:
        return self._peak_bytes

    @property
    def accounted_persistent_bytes(self) -> int:
        """Bytes currently owned by persistent tensor state for this site."""

        return self._current_persistent_bytes()

    def accounted_batch_transient_bytes(
        self, rows: int, pass_number: int
    ) -> int:
        """Conservative concurrent bytes for one hook batch in either pass."""

        checked_rows = _positive_int(rows, "rows")
        if isinstance(pass_number, bool):
            raise OutputJacobianSketchShapeError("pass_number must be 1 or 2.")
        try:
            checked_pass = operator.index(pass_number)
        except (TypeError, ValueError, OverflowError) as error:
            raise OutputJacobianSketchShapeError(
                "pass_number must be 1 or 2."
            ) from error
        if checked_pass not in (1, 2):
            raise OutputJacobianSketchShapeError("pass_number must be 1 or 2.")
        return self._block_transient_bytes(checked_rows, checked_pass)

    def _validate_factors(
        self,
        left: Tensor | None,
        right: Tensor | None,
    ) -> tuple[Tensor, Tensor, int]:
        if (left is None) != (right is None):
            raise OutputJacobianSketchShapeError(
                "inherited_left and inherited_right must be supplied together."
            )
        if left is None:
            return (
                torch.empty((self.spec.out_features, 0), dtype=torch.float32),
                torch.empty((0, self.spec.in_features), dtype=torch.float32),
                0,
            )
        assert right is not None
        left = _validate_dense_float_metadata(left, "inherited_left")
        right = _validate_dense_float_metadata(right, "inherited_right")
        if left.ndim != 2 or right.ndim != 2:
            raise OutputJacobianSketchShapeError(
                "Inherited factors must both be rank-two tensors."
            )
        latent_rank = int(left.shape[1])
        if tuple(left.shape) != (self.spec.out_features, latent_rank):
            raise OutputJacobianSketchShapeError(
                "inherited_left has the wrong output dimension."
            )
        if tuple(right.shape) != (latent_rank, self.spec.in_features):
            raise OutputJacobianSketchShapeError(
                "inherited_right has the wrong latent or input dimension."
            )
        if latent_rank == 0:
            return left.detach(), right.detach(), 0
        out = self.spec.out_features
        inputs = self.spec.in_features
        latent = latent_rank
        left_width = min(out, latent)
        right_width = min(inputs, latent)
        # FP64 factor copies, both reduced QRs, the small product core/SVD,
        # finite-scan masks, and final occupied bases are all conservatively
        # reserved before the first value read or conversion.
        factor_bytes = 8 * latent * (out + inputs)
        qr_bytes = 16 * (
            out * left_width
            + inputs * right_width
            + latent * (left_width + right_width)
        )
        core_bytes = 32 * left_width * right_width
        scan_and_outputs = (
            left.numel()
            + right.numel()
            + 8 * (out + inputs) * min(left_width, right_width)
        )
        self._require_bytes(
            factor_bytes + qr_bytes + core_bytes + scan_and_outputs,
            "inherited-factor rank analysis",
            absolute=True,
        )
        if not bool(torch.isfinite(left.detach()).all()):
            raise OutputJacobianSketchFiniteError("inherited_left is non-finite.")
        if not bool(torch.isfinite(right.detach()).all()):
            raise OutputJacobianSketchFiniteError("inherited_right is non-finite.")
        return left.detach(), right.detach(), latent_rank

    def _occupied_spaces(
        self, left: Tensor, right: Tensor
    ) -> tuple[Tensor, Tensor, float, bool]:
        if self._latent_rank == 0:
            return (
                torch.empty(
                    (self.spec.out_features, 0),
                    dtype=torch.float32,
                    device=self._device,
                ),
                torch.empty(
                    (self.spec.in_features, 0),
                    dtype=torch.float32,
                    device=self._device,
                ),
                0.0,
                True,
            )
        left64 = left.to(device="cpu", dtype=torch.float64)
        right64 = right.to(device="cpu", dtype=torch.float64)
        # Do not use orth(L) and orth(R.T) independently: latent cancellation
        # can make their product rank smaller than either factor.  Reduced
        # factor QRs expose the actual LR spaces through only a small core.
        left_q, left_r = torch.linalg.qr(left64, mode="reduced")
        right_q, right_r = torch.linalg.qr(right64.T, mode="reduced")
        product_core = left_r @ right_r.T
        core_u, singular, core_vh = torch.linalg.svd(
            product_core, full_matrices=False
        )
        if not bool(torch.isfinite(singular).all()):
            raise OutputJacobianSketchFiniteError(
                "Inherited-product rank analysis became non-finite."
            )
        product_frobenius = float(
            torch.linalg.vector_norm(singular).item()
        )
        product_scale = float(
            torch.linalg.vector_norm(left64).item()
            * torch.linalg.vector_norm(right64).item()
        )
        product_zero = product_frobenius <= (
            self.spec.product_zero_rtol * max(product_scale, torch.finfo(torch.float64).tiny)
        )
        if product_zero:
            return (
                torch.empty(
                    (self.spec.out_features, 0),
                    dtype=torch.float32,
                    device=self._device,
                ),
                torch.empty(
                    (self.spec.in_features, 0),
                    dtype=torch.float32,
                    device=self._device,
                ),
                product_frobenius,
                True,
            )
        if singular.numel() == 0 or float(singular[0].item()) <= 0.0:
            raise OutputJacobianSketchShapeError(
                "A nonzero inherited product has no certifiable singular space."
            )
        tolerance = max(
            self.spec.rank_rtol * float(singular[0].item()),
            max(product_core.shape)
            * torch.finfo(torch.float64).eps
            * float(singular[0].item()),
        )
        effective_rank = int((singular > tolerance).sum().item())
        if effective_rank <= 0:
            raise OutputJacobianSketchShapeError(
                "A nonzero inherited product has no certifiable effective rank."
            )
        u = left_q @ core_u[:, :effective_rank]
        v = right_q @ core_vh[:effective_rank].T
        u = _canonicalize_columns(u.contiguous())
        v = _canonicalize_columns(v.contiguous())
        return (
            u.to(device=self._device, dtype=torch.float32),
            v.to(device=self._device, dtype=torch.float32),
            product_frobenius,
            False,
        )

    def _persistent_bytes_for(
        self,
        *,
        probe: bool,
        pass1: bool,
        range_basis: bool,
        pass2: bool,
    ) -> int:
        total = _tensor_bytes(self._u) + _tensor_bytes(self._v)
        if probe:
            total += 4 * self.spec.out_features * self.spec.sketch_width
        if pass1:
            total += 8 * self.spec.in_features * self.spec.sketch_width
        if range_basis:
            total += 4 * self.spec.in_features * (
                self._range_width + self.spec.rank
            )
        if pass2:
            total += 4 * self.spec.out_features * (
                self._range_width + self.spec.rank
            )
        return total

    def _current_persistent_bytes(self) -> int:
        values = (
            self._u,
            self._v,
            getattr(self, "_omega", None),
            getattr(self, "_raw_y", None),
            getattr(self, "_normal_y", None),
            self._q,
            self._control,
            self._b,
            self._random_b,
        )
        return sum(_tensor_bytes(value) for value in values if isinstance(value, Tensor))

    def _require_bytes(
        self, amount: int, operation: str, *, absolute: bool = False
    ) -> None:
        required = int(amount if absolute else self._current_persistent_bytes() + amount)
        self._peak_bytes = max(self._peak_bytes, required)
        if required > self.spec.max_bytes:
            raise OutputJacobianSketchLimitError(
                f"{operation} requires an accounted {required} bytes, exceeding "
                f"max_bytes={self.spec.max_bytes}."
            )

    def _require_phase(self, expected: SketchPhase) -> None:
        if self._phase != expected:
            raise OutputJacobianSketchPhaseError(
                f"Operation requires phase {expected}, not {self._phase}."
            )

    def _inspect_block(self, delta: Any, inputs: Any) -> int:
        delta = _validate_dense_float_metadata(delta, "D")
        inputs = _validate_dense_float_metadata(inputs, "X")
        if delta.ndim != 2 or inputs.ndim != 2:
            raise OutputJacobianSketchShapeError("D and X must be rank-two tensors.")
        if int(delta.shape[0]) <= 0 or int(inputs.shape[0]) != int(delta.shape[0]):
            raise OutputJacobianSketchShapeError(
                "D and X must have the same positive row count."
            )
        if int(delta.shape[1]) != self.spec.out_features:
            raise OutputJacobianSketchShapeError("D has the wrong output width.")
        if int(inputs.shape[1]) != self.spec.in_features:
            raise OutputJacobianSketchShapeError("X has the wrong input width.")
        return int(delta.shape[0])

    def _prepare_block(
        self,
        delta: Tensor,
        inputs: Tensor,
        *,
        pass_number: int,
    ) -> tuple[Tensor, Tensor]:
        self._reserve_block(delta, inputs, pass_number=pass_number)
        if not bool(torch.isfinite(delta.detach()).all()):
            raise OutputJacobianSketchFiniteError("D is non-finite.")
        if not bool(torch.isfinite(inputs.detach()).all()):
            raise OutputJacobianSketchFiniteError("X is non-finite.")
        return self._convert_prevalidated_block(delta, inputs)

    def _reserve_block(
        self,
        delta: Tensor,
        inputs: Tensor,
        *,
        pass_number: int,
    ) -> None:
        rows = self._inspect_block(delta, inputs)
        self._require_bytes(
            self._block_transient_bytes(rows, pass_number),
            f"pass-{'one' if pass_number == 1 else 'two'} batch",
        )

    def _prepare_prevalidated_block(
        self,
        delta: Tensor,
        inputs: Tensor,
        *,
        pass_number: int,
    ) -> tuple[Tensor, Tensor]:
        self._reserve_block(delta, inputs, pass_number=pass_number)
        return self._convert_prevalidated_block(delta, inputs)

    def _convert_prevalidated_block(
        self,
        delta: Tensor,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            delta.detach().to(device=self._device, dtype=torch.float32),
            inputs.detach().to(device=self._device, dtype=torch.float32),
        )

    def make_exact_block_identity(
        self,
        delta: Tensor,
        inputs: Tensor,
        *,
        semantic_identity: str,
        pass_number: int = 1,
    ) -> OutputJacobianBlockIdentity:
        """Hash exact canonical D/X bytes for replayable one-shot inputs.

        Streaming model-hook integrations should construct
        :class:`OutputJacobianBlockIdentity` from stable semantic metadata and
        a wrapper-level compact fingerprint instead, avoiding per-site copies.
        """

        if isinstance(pass_number, bool) or pass_number not in (1, 2):
            raise OutputJacobianSketchShapeError("pass_number must be 1 or 2.")
        rows = self._inspect_block(delta, inputs)
        hash_staging = int(
            delta.numel() * delta.element_size()
            + inputs.numel() * inputs.element_size()
        )
        self._require_bytes(
            self._block_transient_bytes(rows, pass_number) + hash_staging,
            "exact block identity hashing",
        )
        if not bool(torch.isfinite(delta.detach()).all()):
            raise OutputJacobianSketchFiniteError("D is non-finite.")
        if not bool(torch.isfinite(inputs.detach()).all()):
            raise OutputJacobianSketchFiniteError("X is non-finite.")
        delta_hash = _tensor_sha256(delta)
        inputs_hash = _tensor_sha256(inputs)
        numeric_hash = _json_sha256(
            {
                "schema": f"{OUTPUT_JACOBIAN_SKETCH_SCHEMA}:exact-block-v1",
                "delta_sha256": delta_hash,
                "inputs_sha256": inputs_hash,
            }
        )
        return OutputJacobianBlockIdentity(
            semantic_identity=semantic_identity,
            numeric_fingerprint_sha256=numeric_hash,
            delta_sha256=delta_hash,
            inputs_sha256=inputs_hash,
        )

    @staticmethod
    def _checked_block_identity(
        block_identity: Any,
    ) -> OutputJacobianBlockIdentity:
        if not isinstance(block_identity, OutputJacobianBlockIdentity):
            raise OutputJacobianSketchShapeError(
                "block_identity must be an OutputJacobianBlockIdentity."
            )
        return block_identity

    @staticmethod
    def _extend_identity_chain(
        chain: Any, identity: OutputJacobianBlockIdentity
    ) -> None:
        chain.update(bytes.fromhex(identity.sha256()))

    def _normal_blocks(self, delta: Tensor, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if int(self._u.shape[1]) > 0:
            normal_delta = delta - (delta @ self._u) @ self._u.T
        else:
            normal_delta = delta
        if int(self._v.shape[1]) > 0:
            normal_inputs = inputs - (inputs @ self._v) @ self._v.T
        else:
            normal_inputs = inputs
        return normal_delta, normal_inputs

    def _block_transient_bytes(self, rows: int, pass_number: int) -> int:
        out = self.spec.out_features
        inputs = self.spec.in_features
        u_rank = int(self._u.shape[1])
        v_rank = int(self._v.shape[1])
        # Conservative peak: FP32 conversions, projected blocks and both sides
        # of each subtraction may coexist briefly.  This intentionally
        # overcounts aliases when caller tensors already are CPU FP32.
        base = 4 * rows * (3 * out + 3 * inputs + u_rank + v_rank)
        if pass_number == 1:
            products = 4 * (
                rows * self.spec.sketch_width
                + self.spec.in_features * self.spec.sketch_width
            )
        else:
            products = 4 * (
                rows * (self._range_width + self.spec.rank)
                + self.spec.out_features * max(self._range_width, self.spec.rank)
            )
        return base + products

    def add_pass1(
        self,
        delta: Tensor,
        inputs: Tensor,
        *,
        block_identity: OutputJacobianBlockIdentity,
    ) -> None:
        """Accumulate range and Hutchinson probes from one finite row batch."""

        self._require_phase("pass1")
        if self._pass1_prevalidated:
            raise OutputJacobianSketchPhaseError(
                "Validated and deferred-identity pass-one ingestion cannot be mixed."
            )
        identity = self._checked_block_identity(block_identity)
        d, x = self._prepare_block(delta, inputs, pass_number=1)
        self._accumulate_pass1(d, x)
        if not bool(torch.isfinite(self._raw_y).all()):
            raise OutputJacobianSketchFiniteError(
                "The raw pass-one accumulator became non-finite."
            )
        if not bool(torch.isfinite(self._normal_y).all()):
            raise OutputJacobianSketchFiniteError(
                "The normal pass-one accumulator became non-finite."
            )
        self._extend_identity_chain(self._pass1_identity_chain, identity)

    def add_prevalidated_pass1(self, delta: Tensor, inputs: Tensor) -> None:
        """Accumulate a block whose full wrapper tensors are validated later.

        This hot-path entry point is reserved for integrations which authenticate
        full-wrapper finiteness and ordered numeric fingerprints in one bounded
        pass-level transfer.  Shape, dtype, device, byte-ceiling, and phase checks
        still occur before accumulation.  It cannot be mixed with per-block
        identities.
        """

        self._require_phase("pass1")
        if self._pass1_blocks and not self._pass1_prevalidated:
            raise OutputJacobianSketchPhaseError(
                "Validated and per-block-identity pass-one ingestion cannot be mixed."
            )
        self._pass1_prevalidated = True
        d, x = self._prepare_prevalidated_block(
            delta,
            inputs,
            pass_number=1,
        )
        self._accumulate_pass1(d, x)

    def _accumulate_pass1(self, d: Tensor, x: Tensor) -> None:
        rows = int(d.shape[0])
        normal_d, normal_x = self._normal_blocks(d, x)
        self._raw_y.add_(x.T @ (d @ self._omega))
        self._normal_y.add_(normal_x.T @ (normal_d @ self._omega))
        self._pass1_samples += rows
        self._pass1_blocks += 1

    def bind_prevalidated_identity_chain(
        self, pass_number: int, chain_sha256: str
    ) -> None:
        """Bind a deferred, ordered block-identity chain before a transition."""

        digest = _canonical_sha256(chain_sha256, "chain_sha256")
        if isinstance(pass_number, bool) or pass_number not in (1, 2):
            raise OutputJacobianSketchShapeError("pass_number must be 1 or 2.")
        expected_phase: SketchPhase = "pass1" if pass_number == 1 else "pass2"
        self._require_phase(expected_phase)
        prevalidated = (
            self._pass1_prevalidated if pass_number == 1 else self._pass2_prevalidated
        )
        block_count = self._pass1_blocks if pass_number == 1 else self._pass2_blocks
        if not prevalidated or block_count <= 0:
            raise OutputJacobianSketchPhaseError(
                "A deferred identity chain requires nonempty prevalidated ingestion."
            )
        if pass_number == 1:
            if self._pass1_identity_sha256:
                raise OutputJacobianSketchPhaseError(
                    "The pass-one deferred identity chain is already bound."
                )
            self._pass1_identity_sha256 = digest
        else:
            if self._pass2_identity_sha256:
                raise OutputJacobianSketchPhaseError(
                    "The pass-two deferred identity chain is already bound."
                )
            self._pass2_identity_sha256 = digest

    def _complement_basis(
        self,
        initial: Tensor | None,
        columns: int,
        against: Sequence[Tensor],
        label: str,
    ) -> Tensor:
        dimension = self.spec.in_features
        bases = [
            basis.to(device="cpu", dtype=torch.float64)
            for basis in against
            if basis.numel() > 0
        ]
        selected = torch.empty((dimension, 0), dtype=torch.float64)

        def project(value: Tensor) -> Tensor:
            result = value
            for basis in (*bases, selected):
                if int(basis.shape[1]) > 0:
                    result = result - basis @ (basis.T @ result)
            return result

        if initial is not None and initial.numel() > 0:
            candidate = project(initial.to(device="cpu", dtype=torch.float64))
            q, r = torch.linalg.qr(candidate, mode="reduced")
            diagonal = torch.diagonal(r).abs()
            if diagonal.numel() > 0:
                scale = float(diagonal.max().item())
                tolerance = max(
                    self.spec.rank_rtol * scale,
                    max(candidate.shape) * torch.finfo(torch.float64).eps * scale,
                )
                keep = diagonal > tolerance
                selected = q[:, keep]
        attempt = 0
        while int(selected.shape[1]) < columns and attempt < 16:
            needed = columns - int(selected.shape[1])
            random = _rademacher(
                dimension,
                needed + min(4, dimension),
                _derived_seed(self.spec.seed, label, attempt),
            ).to(torch.float64)
            candidate = project(random)
            q, r = torch.linalg.qr(candidate, mode="reduced")
            diagonal = torch.diagonal(r).abs()
            if diagonal.numel() > 0:
                scale = float(diagonal.max().item())
                tolerance = max(
                    self.spec.rank_rtol * scale,
                    max(candidate.shape) * torch.finfo(torch.float64).eps * scale,
                )
                addition = q[:, diagonal > tolerance][:, :needed]
                if addition.numel() > 0:
                    selected = torch.cat((selected, addition), dim=1)
            attempt += 1
        if int(selected.shape[1]) != columns:
            raise OutputJacobianSketchShapeError(
                f"Could not construct {columns} independent {label} directions."
            )
        # A final projection against only the fixed occupied spaces, followed
        # by QR, controls drift.  ``project`` also removes the live ``selected``
        # basis and is therefore intentionally not used on the basis itself.
        final = selected
        for basis in bases:
            final = final - basis @ (basis.T @ final)
        selected, _ = torch.linalg.qr(final, mode="reduced")
        selected = _canonicalize_columns(selected[:, :columns])
        if not bool(torch.isfinite(selected).all()):
            raise OutputJacobianSketchFiniteError(
                f"The {label} basis became non-finite."
            )
        return selected.to(device=self._device, dtype=torch.float32)

    def finish_pass1(self) -> None:
        """Finalize Hutchinson estimates and the range/control bases."""

        self._require_phase("pass1")
        if self._pass1_blocks == 0:
            raise OutputJacobianSketchPhaseError(
                "At least one pass-one batch is required."
            )
        if self._pass1_prevalidated:
            if not self._pass1_identity_sha256:
                raise OutputJacobianSketchPhaseError(
                    "Pass one lacks its deferred block-identity chain."
                )
        else:
            self._pass1_identity_sha256 = self._pass1_identity_chain.hexdigest()
        transition_bytes = (
            64 * self.spec.in_features * self.spec.sketch_width
            + 32
            * self.spec.in_features
            * (self.spec.rank + int(self._v.shape[1]))
            + 32 * self.spec.sketch_width * self.spec.sketch_width
        )
        self._require_bytes(transition_bytes, "pass-one FP64 QR transition")
        raw_y_cpu = self._raw_y.detach().to(device="cpu").contiguous()
        normal_y_cpu = self._normal_y.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(raw_y_cpu).all()):
            raise OutputJacobianSketchFiniteError(
                "The raw pass-one accumulator became non-finite."
            )
        if not bool(torch.isfinite(normal_y_cpu).all()):
            raise OutputJacobianSketchFiniteError(
                "The normal pass-one accumulator became non-finite."
            )
        self._raw_frobenius_sq = float(
            raw_y_cpu.to(dtype=torch.float64).square().sum().item()
            / self.spec.sketch_width
        )
        self._normal_frobenius_sq = float(
            normal_y_cpu.to(dtype=torch.float64).square().sum().item()
            / self.spec.sketch_width
        )
        self._pass1_normal_hash = _tensor_sha256(normal_y_cpu)
        self._q = self._complement_basis(
            self._normal_y,
            self._range_width,
            (self._v,),
            "range-basis-completion",
        )
        self._control = self._complement_basis(
            None,
            self.spec.rank,
            (self._v,),
            "random-complement-control",
        )

        # Release the first-pass-only matrices before allocating pass two.
        self._omega = None  # type: ignore[assignment]
        self._raw_y = None  # type: ignore[assignment]
        self._normal_y = None  # type: ignore[assignment]
        prospective = self._persistent_bytes_for(
            probe=False,
            pass1=False,
            range_basis=True,
            pass2=True,
        )
        self._require_bytes(prospective, "pass-two allocation", absolute=True)
        self._b = torch.zeros(
            (self.spec.out_features, self._range_width),
            dtype=torch.float32,
            device=self._device,
        )
        self._random_b = torch.zeros(
            (self.spec.out_features, self.spec.rank),
            dtype=torch.float32,
            device=self._device,
        )
        self._phase = "pass2"

    def add_pass2(
        self,
        delta: Tensor,
        inputs: Tensor,
        *,
        block_identity: OutputJacobianBlockIdentity,
    ) -> None:
        """Replay one batch to accumulate guided and random-control cores."""

        self._require_phase("pass2")
        if self._pass2_prevalidated:
            raise OutputJacobianSketchPhaseError(
                "Validated and deferred-identity pass-two ingestion cannot be mixed."
            )
        identity = self._checked_block_identity(block_identity)
        d, x = self._prepare_block(delta, inputs, pass_number=2)
        self._accumulate_pass2(d, x)
        if not bool(torch.isfinite(self._b).all()):
            raise OutputJacobianSketchFiniteError(
                "The guided pass-two accumulator became non-finite."
            )
        if not bool(torch.isfinite(self._random_b).all()):
            raise OutputJacobianSketchFiniteError(
                "The random pass-two accumulator became non-finite."
            )
        self._extend_identity_chain(self._pass2_identity_chain, identity)

    def add_prevalidated_pass2(self, delta: Tensor, inputs: Tensor) -> None:
        """Replay a block under pass-level finite/identity validation."""

        self._require_phase("pass2")
        if self._pass2_blocks and not self._pass2_prevalidated:
            raise OutputJacobianSketchPhaseError(
                "Validated and per-block-identity pass-two ingestion cannot be mixed."
            )
        self._pass2_prevalidated = True
        d, x = self._prepare_prevalidated_block(
            delta,
            inputs,
            pass_number=2,
        )
        self._accumulate_pass2(d, x)

    def _accumulate_pass2(self, d: Tensor, x: Tensor) -> None:
        rows = int(d.shape[0])
        normal_d, normal_x = self._normal_blocks(d, x)
        assert self._q is not None
        assert self._control is not None
        assert self._b is not None
        assert self._random_b is not None
        self._b.add_(normal_d.T @ (normal_x @ self._q))
        self._random_b.add_(normal_d.T @ (normal_x @ self._control))
        self._pass2_samples += rows
        self._pass2_blocks += 1

    def finalize(self) -> OutputJacobianSketchResult:
        """Return canonical rank-by-input guided and random control rows."""

        self._require_phase("pass2")
        if (
            self._pass2_samples != self._pass1_samples
            or self._pass2_blocks != self._pass1_blocks
        ):
            raise OutputJacobianSketchPhaseError(
                "Pass two must replay the same row and block counts as pass one."
            )
        if self._pass2_prevalidated:
            if not self._pass2_identity_sha256:
                raise OutputJacobianSketchPhaseError(
                    "Pass two lacks its deferred block-identity chain."
                )
            pass2_identity_sha256 = self._pass2_identity_sha256
        else:
            pass2_identity_sha256 = self._pass2_identity_chain.hexdigest()
        if pass2_identity_sha256 != self._pass1_identity_sha256:
            raise OutputJacobianSketchPhaseError(
                "Pass two block identities or order do not authenticate against "
                "pass one."
            )
        assert self._q is not None
        assert self._control is not None
        assert self._b is not None
        assert self._random_b is not None
        # Conservative simultaneous live set for the FP64 B input, reduced
        # SVD outputs, Q conversion/product, and both canonical FP32 row sets.
        finalize_bytes = (
            16 * self.spec.out_features * self._range_width
            + 8 * self._range_width * self._range_width
            + 8 * self._range_width
            + 8
            * self.spec.in_features
            * (self._range_width + self.spec.rank)
            + 12 * self.spec.rank * self.spec.in_features
        )
        self._require_bytes(finalize_bytes, "final FP64 SVD")
        b_cpu = self._b.detach().to(device="cpu").contiguous()
        random_b_cpu = self._random_b.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(b_cpu).all()):
            raise OutputJacobianSketchFiniteError(
                "The guided pass-two accumulator became non-finite."
            )
        if not bool(torch.isfinite(random_b_cpu).all()):
            raise OutputJacobianSketchFiniteError(
                "The random pass-two accumulator became non-finite."
            )
        _, singular, vh = torch.linalg.svd(
            b_cpu.to(dtype=torch.float64), full_matrices=False
        )
        if not bool(torch.isfinite(singular).all()):
            raise OutputJacobianSketchFiniteError(
                "The final guided SVD produced non-finite values."
            )
        if singular.numel() == 0 or float(singular[0].item()) <= 0.0:
            raise OutputJacobianSketchRankError(
                "The normal output Jacobian has zero effective rank."
            )
        leading = float(singular[0].item())
        rank_tolerance = max(
            self.spec.rank_rtol * leading,
            max(self._b.shape)
            * torch.finfo(torch.float64).eps
            * leading,
        )
        effective_rank = int((singular > rank_tolerance).sum().item())
        if effective_rank < self.spec.rank:
            raise OutputJacobianSketchRankError(
                "The normal output Jacobian effective rank is below the "
                "requested rank."
            )
        if self.spec.rank < effective_rank:
            boundary_gap = float(
                (singular[self.spec.rank - 1] - singular[self.spec.rank]).abs().item()
            )
            if boundary_gap <= self.spec.rank_rtol * leading:
                raise OutputJacobianSketchRankError(
                    "The requested singular-rank boundary is numerically "
                    "degenerate."
                )

        q64 = self._q.to(device="cpu", dtype=torch.float64)
        directions = q64 @ vh[: self.spec.rank].T
        # SVD vectors are not deterministic inside a repeated-singular-value
        # cluster.  Replace each wholly selected exact cluster with a
        # coordinate-canonical basis of the same full-input subspace.
        degeneracy_tolerance = rank_tolerance
        cluster_start = 0
        while cluster_start < self.spec.rank:
            cluster_end = cluster_start + 1
            while (
                cluster_end < self.spec.rank
                and float(
                    (
                        singular[cluster_end - 1] - singular[cluster_end]
                    ).abs().item()
                )
                <= degeneracy_tolerance
            ):
                cluster_end += 1
            if cluster_end - cluster_start > 1:
                cluster = q64 @ vh[cluster_start:cluster_end].T
                directions[:, cluster_start:cluster_end] = (
                    _coordinate_canonical_subspace(cluster)
                )
            cluster_start = cluster_end
        guided_rows, guided_pivots = _canonicalize_rows(
            directions.T.to(device=self._device, dtype=torch.float32).contiguous()
        )
        random_rows, random_pivots = _canonicalize_rows(
            self._control.T.contiguous()
        )

        guided_energy = float(
            singular[: self.spec.rank].square().sum().item()
        )
        random_energy = float(
            random_b_cpu.to(dtype=torch.float64).square().sum().item()
        )
        raw_scale = max(
            self._raw_frobenius_sq,
            torch.finfo(torch.float64).tiny,
        )
        normal_scale = max(
            self._normal_frobenius_sq,
            torch.finfo(torch.float64).tiny,
        )
        normal_fraction = (
            self._normal_frobenius_sq / raw_scale
            if self._raw_frobenius_sq > 0.0
            else 0.0
        )
        guided_fraction = (
            guided_energy / normal_scale
            if self._normal_frobenius_sq > 0.0
            else 0.0
        )
        random_fraction = (
            random_energy / normal_scale
            if self._normal_frobenius_sq > 0.0
            else 0.0
        )
        ratio_floor = max(normal_scale * 1e-30, torch.finfo(torch.float64).tiny)
        guided_over_random = guided_energy / max(random_energy, ratio_floor)
        normal_gate = normal_fraction >= self.spec.min_normal_fraction
        guided_gate = guided_fraction >= self.spec.min_guided_capture_fraction
        ratio_gate = (
            guided_over_random >= self.spec.min_guided_over_random_ratio
        )

        report_payload: dict[str, Any] = {
            "schema": OUTPUT_JACOBIAN_SKETCH_SCHEMA,
            "spec_sha256": self.spec.sha256(),
            "accumulator_device": str(self._device),
            "sample_count": self._pass1_samples,
            "block_count": self._pass1_blocks,
            "probe_count": self.spec.sketch_width,
            "range_basis_width": self._range_width,
            "normal_effective_rank": effective_rank,
            "selected_singular_values": tuple(
                float(value) for value in singular[: self.spec.rank].tolist()
            ),
            "inherited_latent_rank": self._latent_rank,
            "occupied_output_rank": int(self._u.shape[1]),
            "occupied_input_rank": int(self._v.shape[1]),
            "inherited_product_frobenius": self._product_frobenius,
            "inherited_product_numerically_zero": self._product_zero,
            "raw_frobenius_sq_estimate": self._raw_frobenius_sq,
            "normal_frobenius_sq_estimate": self._normal_frobenius_sq,
            "normal_fraction": normal_fraction,
            "guided_capture_energy": guided_energy,
            "guided_capture_fraction": guided_fraction,
            "random_capture_energy": random_energy,
            "random_capture_fraction": random_fraction,
            "guided_over_random_ratio": guided_over_random,
            "minimum_normal_fraction": self.spec.min_normal_fraction,
            "minimum_guided_capture_fraction": (
                self.spec.min_guided_capture_fraction
            ),
            "minimum_guided_over_random_ratio": (
                self.spec.min_guided_over_random_ratio
            ),
            "normal_fraction_gate_passed": normal_gate,
            "guided_capture_gate_passed": guided_gate,
            "guided_over_random_gate_passed": ratio_gate,
            "gate_passed": normal_gate and guided_gate and ratio_gate,
            "accounted_peak_bytes": self._peak_bytes,
            "max_bytes": self.spec.max_bytes,
            "probe_sha256": self._probe_hash,
            "inherited_left_sha256": self._left_hash,
            "inherited_right_sha256": self._right_hash,
            "occupied_output_basis_sha256": _tensor_sha256(self._u),
            "occupied_input_basis_sha256": _tensor_sha256(self._v),
            "pass1_normal_sketch_sha256": self._pass1_normal_hash,
            "range_basis_sha256": _tensor_sha256(self._q),
            "random_control_sha256": _tensor_sha256(self._control),
            "pass2_guided_core_sha256": _tensor_sha256(b_cpu),
            "pass2_random_core_sha256": _tensor_sha256(random_b_cpu),
            "guided_right_rows_sha256": _tensor_sha256(guided_rows),
            "random_right_rows_sha256": _tensor_sha256(random_rows),
            "pass1_block_identity_chain_sha256": (
                self._pass1_identity_sha256
            ),
            "pass2_block_identity_chain_sha256": pass2_identity_sha256,
            "block_identity_replay_authenticated": True,
            "guided_sign_pivots": guided_pivots,
            "random_sign_pivots": random_pivots,
        }
        report_payload["report_sha256"] = _json_sha256(report_payload)
        report = OutputJacobianSketchReport(**report_payload)
        self._phase = "finalized"
        return OutputJacobianSketchResult(
            guided_right_rows=guided_rows,
            random_right_rows=random_rows,
            report=report,
        )


def sketch_output_jacobian(
    blocks: Sequence[tuple[Tensor, Tensor]],
    spec: OutputJacobianSketchSpec,
    inherited_left: Tensor | None = None,
    inherited_right: Tensor | None = None,
    *,
    accumulator_device: str | torch.device | None = None,
) -> OutputJacobianSketchResult:
    """One-shot convenience wrapper which replays a finite block sequence."""

    if not isinstance(blocks, Sequence) or len(blocks) == 0:
        raise OutputJacobianSketchShapeError(
            "blocks must be a non-empty replayable sequence."
        )
    accumulator = TwoPassOutputJacobianSketch(
        spec,
        inherited_left=inherited_left,
        inherited_right=inherited_right,
        accumulator_device=accumulator_device,
    )
    for block_index, (delta, inputs) in enumerate(blocks):
        identity = accumulator.make_exact_block_identity(
            delta,
            inputs,
            semantic_identity=f"one-shot:block:{block_index}",
        )
        accumulator.add_pass1(delta, inputs, block_identity=identity)
    accumulator.finish_pass1()
    for block_index, (delta, inputs) in enumerate(blocks):
        identity = accumulator.make_exact_block_identity(
            delta,
            inputs,
            semantic_identity=f"one-shot:block:{block_index}",
            pass_number=2,
        )
        accumulator.add_pass2(delta, inputs, block_identity=identity)
    return accumulator.finalize()


__all__ = [
    "OUTPUT_JACOBIAN_SKETCH_SCHEMA",
    "MINIMUM_GUIDED_OVER_RANDOM_RATIO",
    "MINIMUM_NORMAL_FRACTION",
    "OutputJacobianSketchError",
    "OutputJacobianBlockIdentity",
    "OutputJacobianSketchFiniteError",
    "OutputJacobianSketchLimitError",
    "OutputJacobianSketchPhaseError",
    "OutputJacobianSketchRankError",
    "OutputJacobianSketchReport",
    "OutputJacobianSketchResult",
    "OutputJacobianSketchShapeError",
    "OutputJacobianSketchSpec",
    "TwoPassOutputJacobianSketch",
    "sketch_output_jacobian",
]
