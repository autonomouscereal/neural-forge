"""Exact multiple-choice byte planning for CREVF tensor/bundle sites.

The planner consumes real authenticated main tensors, correction tensors, and
bounded auxiliary blobs, not idealized ``numel * bits`` estimates. Every
option therefore pays every complete record, hash, and artifact envelope.
The artifact header/trailer and caller-declared shared checkpoint files are
also counted before a plan can pass.

This module is intentionally sensitivity-agnostic.  ``loss_units`` are sealed
integer scores supplied by a later causal/Hessian component.  Integer scores
avoid platform-dependent floating-point tie decisions.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .hf_lfm2_causal_residual_entropy_value_fold import (
    CREVF_ARTIFACT_FIXED_BYTES,
    CREVF_RECORD_ENVELOPE_BYTES,
    CrevfBudgetError,
    CrevfError,
    CrevfAuxiliaryMetadata,
    CrevfReductionTarget,
    CrevfTensorMetadata,
    crevf_decode_transient_group,
    encode_crevf_artifact,
    inspect_crevf_auxiliary,
    inspect_crevf_tensor,
)


_QUERY_TOKENS = frozenset({"q", "query", "qproj", "q_proj", "wq"})
_KEY_TOKENS = frozenset({"k", "key", "kproj", "k_proj", "wk", "kv", "kvproj"})
_NORM_TOKENS = frozenset(
    {
        "norm",
        "normalization",
        "layernorm",
        "rmsnorm",
        "groupnorm",
        "batchnorm",
        "instancenorm",
        "inputlayernorm",
        "postattentionlayernorm",
        "ln",
        "ln1",
        "ln2",
        "lnf",
    }
)
_EXACT_ROLE_NAMES = frozenset(
    {
        "control_anchor",
        "exact_anchor",
        "chat_control_anchor",
        "special_token_anchor",
    }
)
_SEMANTIC_FLOORS = {
    "other": 4,
    "attention_query": 7,
    "attention_key": 7,
    "attention_query_key_fused": 7,
    "normalization": 16,
    "control_anchor": 16,
}


def _canonical_identifier(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise CrevfError(f"{name} must be nonempty canonical NFC text.")
    return value


def _nonnegative_int(value: int, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise CrevfError(f"{name} must be a {qualifier} integer.")
    return value


def _role_tokens(role: str) -> frozenset[str]:
    normalized = role.casefold()
    tokens = frozenset(token for token in re.split(r"[^a-z0-9]+", normalized) if token)
    expanded = set(tokens)
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    expanded.add(compact)
    return frozenset(expanded)


def _tokens_require_exact_norm(tokens: frozenset[str]) -> bool:
    return bool(tokens & _NORM_TOKENS)


def role_requires_exact_bf16(role: str) -> bool:
    """Return the non-bypassable exact floor for norms/control anchors."""

    normalized = _canonical_identifier(role, "site role").casefold()
    tokens = _role_tokens(role)
    return (
        _tokens_require_exact_norm(tokens)
        or normalized in _EXACT_ROLE_NAMES
        or ({"control", "anchor"} <= tokens)
        or "controlanchor" in tokens
        or "exactanchor" in tokens
    )


def role_minimum_integer_width(role: str) -> int:
    """Return the hard Q/K precision floor, independent of supplied loss."""

    if role_requires_exact_bf16(role):
        return 16
    tokens = _role_tokens(role)
    normalized = role.casefold()
    if (
        tokens & _QUERY_TOKENS
        or tokens & _KEY_TOKENS
        or "q_proj" in normalized
        or "k_proj" in normalized
        or "query_proj" in normalized
        or "key_proj" in normalized
        or "qkv" in normalized
        or "wqkv" in normalized
        or "c_attn" in normalized
        or "query_key" in normalized
        or "querykeyvalue" in normalized
        or "w_pack" in normalized
        or "in_proj" in normalized
    ):
        return 7
    return 4


def tensor_identity_minimum_width(tensor_name: str) -> int:
    """Derive an independent floor from the sealed canonical tensor path."""

    normalized = _canonical_identifier(tensor_name, "tensor name").casefold()
    tokens = _role_tokens(tensor_name)
    if (
        _tokens_require_exact_norm(tokens)
        or normalized in _EXACT_ROLE_NAMES
        or ({"control", "anchor"} <= tokens)
        or "controlanchor" in tokens
        or "exactanchor" in tokens
    ):
        return 16
    if (
        tokens & _QUERY_TOKENS
        or tokens & _KEY_TOKENS
        or "q_proj" in normalized
        or "k_proj" in normalized
        or "query_proj" in normalized
        or "key_proj" in normalized
        or "qkv" in normalized
        or "wqkv" in normalized
        or "c_attn" in normalized
        or "query_key" in normalized
        or "querykeyvalue" in normalized
        or "w_pack" in normalized
        or "in_proj" in normalized
    ):
        return 7
    return 4


def semantic_minimum_integer_width(semantic_kind: str) -> int:
    """Return the adapter/catalog-sealed semantic precision floor."""

    normalized = _canonical_identifier(
        semantic_kind,
        "semantic_kind",
    ).casefold()
    try:
        return _SEMANTIC_FLOORS[normalized]
    except KeyError as exc:
        raise CrevfError(
            f"semantic_kind must be one of {tuple(sorted(_SEMANTIC_FLOORS))}."
        ) from exc


@dataclass(frozen=True, slots=True)
class CrevfSiteOption:
    """One atomic main-tensor plus correction/auxiliary bundle alternative."""

    option_id: str
    tensor_blob: bytes = field(repr=False)
    loss_units: int
    correction_tensor_blobs: tuple[bytes, ...] = field(default=(), repr=False)
    auxiliary_blobs: tuple[bytes, ...] = field(default=(), repr=False)
    metadata: CrevfTensorMetadata = field(init=False, repr=False)
    tensor_metadata: tuple[CrevfTensorMetadata, ...] = field(
        init=False,
        repr=False,
    )
    auxiliary_metadata: tuple[CrevfAuxiliaryMetadata, ...] = field(
        init=False,
        repr=False,
    )
    record_names: tuple[str, ...] = field(init=False)
    record_sha256s: tuple[tuple[str, str], ...] = field(init=False)
    bundle_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _canonical_identifier(self.option_id, "option_id")
        _nonnegative_int(self.loss_units, "loss_units")
        if not isinstance(self.tensor_blob, bytes):
            raise CrevfError("tensor_blob must be immutable bytes.")
        correction_blobs = tuple(self.correction_tensor_blobs)
        auxiliary_blobs = tuple(self.auxiliary_blobs)
        if any(not isinstance(blob, bytes) for blob in correction_blobs):
            raise CrevfError("correction_tensor_blobs must contain immutable bytes.")
        if any(not isinstance(blob, bytes) for blob in auxiliary_blobs):
            raise CrevfError("auxiliary_blobs must contain immutable bytes.")
        object.__setattr__(self, "correction_tensor_blobs", correction_blobs)
        object.__setattr__(self, "auxiliary_blobs", auxiliary_blobs)
        tensor_metadata = tuple(
            inspect_crevf_tensor(blob)
            for blob in (self.tensor_blob, *correction_blobs)
        )
        auxiliary_metadata = tuple(
            inspect_crevf_auxiliary(blob) for blob in auxiliary_blobs
        )
        metadata = tensor_metadata[0]
        records = tuple(
            [
                *(
                    (value.name, blob)
                    for value, blob in zip(
                        tensor_metadata,
                        (self.tensor_blob, *correction_blobs),
                        strict=True,
                    )
                ),
                *(
                    (value.name, blob)
                    for value, blob in zip(
                        auxiliary_metadata,
                        auxiliary_blobs,
                        strict=True,
                    )
                ),
            ]
        )
        names = tuple(name for name, _blob in records)
        if len(names) != len(set(names)):
            raise CrevfError(
                "A CREVF bundle alternative cannot repeat a record name."
            )
        ordered_records = tuple(sorted(records, key=lambda value: value[0]))
        record_sha256s = tuple(
            (name, hashlib.sha256(blob).hexdigest())
            for name, blob in ordered_records
        )
        digest = hashlib.sha256()
        digest.update(b"neural-forge-crevf-site-bundle-v1\0")
        for name, blob in ordered_records:
            raw_name = name.encode("utf-8")
            digest.update(len(raw_name).to_bytes(8, "little"))
            digest.update(raw_name)
            digest.update(len(blob).to_bytes(8, "little"))
            digest.update(hashlib.sha256(blob).digest())
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "tensor_metadata", tensor_metadata)
        object.__setattr__(self, "auxiliary_metadata", auxiliary_metadata)
        object.__setattr__(self, "record_names", tuple(sorted(names)))
        object.__setattr__(self, "record_sha256s", record_sha256s)
        object.__setattr__(self, "bundle_sha256", digest.hexdigest())

    @property
    def encoded_record_bytes(self) -> int:
        return sum(
            CREVF_RECORD_ENVELOPE_BYTES + len(blob)
            for blob in self.tensor_blobs + self.auxiliary_blobs
        )

    @property
    def tensor_blobs(self) -> tuple[bytes, ...]:
        return (self.tensor_blob, *self.correction_tensor_blobs)

    @property
    def persistent_encoded_bytes(self) -> int:
        return sum(
            value.persistent_encoded_bytes for value in self.tensor_metadata
        ) + sum(
            value.persistent_encoded_bytes
            for value in self.auxiliary_metadata
        )

    @property
    def decoded_full_state_bytes(self) -> int:
        return sum(
            value.decoded_full_state_bytes for value in self.tensor_metadata
        ) + sum(
            value.decoded_full_state_bytes
            for value in self.auxiliary_metadata
        )

    @property
    def decoded_code_transient_bytes(self) -> int:
        return sum(
            value.decoded_code_transient_bytes for value in self.tensor_metadata
        )

    @property
    def maximum_tensor_decode_transient_bytes(self) -> int:
        return max(
            (value.decoded_code_transient_bytes for value in self.tensor_metadata),
            default=0,
        )

    @property
    def decode_transient_group(self) -> str:
        groups = {
            crevf_decode_transient_group(value.name)
            for value in self.tensor_metadata
            if value.decoded_code_transient_bytes
        }
        if len(groups) > 1:
            raise CrevfError(
                "An atomic CREVF bundle may not span multiple nonzero "
                "layer-transient groups."
            )
        return (
            next(iter(groups))
            if groups
            else crevf_decode_transient_group(self.metadata.name)
        )

    @property
    def runtime_packed_bytes(self) -> int:
        """Backward-compatible alias for persistent encoded bytes."""

        return self.persistent_encoded_bytes

    @property
    def minimum_bit_width(self) -> int:
        return min(value.minimum_bit_width for value in self.tensor_metadata)

    @property
    def exact_bf16(self) -> bool:
        return all(value.exact_only for value in self.tensor_metadata)

    @property
    def record_identity_floors_satisfied(self) -> bool:
        for metadata in self.tensor_metadata:
            floor = max(
                role_minimum_integer_width(metadata.role),
                tensor_identity_minimum_width(metadata.name),
            )
            if floor == 16:
                if not metadata.exact_only:
                    return False
            elif metadata.minimum_bit_width < floor:
                return False
        return True


@dataclass(frozen=True, slots=True)
class CrevfSite:
    """A tensor site with mutually exclusive encoded alternatives."""

    site_id: str
    role: str
    options: tuple[CrevfSiteOption, ...]
    semantic_kind: str = "other"

    def __post_init__(self) -> None:
        _canonical_identifier(self.site_id, "site_id")
        _canonical_identifier(self.role, "site role")
        semantic_minimum_integer_width(self.semantic_kind)
        options = tuple(self.options)
        object.__setattr__(self, "options", options)
        if not options:
            raise CrevfError(f"CREVF site {self.site_id!r} has no options.")
        identity_floor = tensor_identity_minimum_width(self.site_id)
        declared_floor = role_minimum_integer_width(self.role)
        if declared_floor < identity_floor:
            raise CrevfError(
                f"Site role {self.role!r} understates the sealed tensor identity "
                f"floor for {self.site_id!r}."
            )
        option_ids = tuple(option.option_id for option in options)
        if len(option_ids) != len(set(option_ids)):
            raise CrevfError(f"CREVF site {self.site_id!r} repeats an option_id.")
        first = options[0].metadata
        site_transient_group = crevf_decode_transient_group(self.site_id)
        for option in options:
            metadata = option.metadata
            if option.decode_transient_group != site_transient_group:
                raise CrevfError(
                    "A CREVF bundle's nonzero transient tensors must remain "
                    "inside its main site's layer-transient group."
                )
            if metadata.name != self.site_id or metadata.role != self.role:
                raise CrevfError(
                    f"Option {option.option_id!r} metadata does not match "
                    f"site {self.site_id!r}."
                )
            if (
                metadata.shape != first.shape
                or metadata.source_numel != first.source_numel
            ):
                raise CrevfError(
                    f"Options for site {self.site_id!r} disagree on tensor geometry."
                )

    @property
    def eligible_options(self) -> tuple[CrevfSiteOption, ...]:
        minimum_width = max(
            role_minimum_integer_width(self.role),
            tensor_identity_minimum_width(self.site_id),
            semantic_minimum_integer_width(self.semantic_kind),
        )
        exact_required = minimum_width == 16
        eligible = tuple(
            sorted(
                (
                    option
                    for option in self.options
                    if (
                        option.record_identity_floors_satisfied
                        and (
                            option.exact_bf16
                            if exact_required
                            else option.minimum_bit_width >= minimum_width
                        )
                    )
                ),
                key=lambda option: option.option_id,
            )
        )
        if not eligible:
            floor = "exact BF16" if exact_required else f"{minimum_width}-bit"
            raise CrevfBudgetError(
                f"Site {self.site_id!r} has no option satisfying its hard {floor} floor."
            )
        return eligible


@dataclass(frozen=True, slots=True)
class CrevfPlanChoice:
    site_id: str
    role: str
    semantic_kind: str
    option_id: str
    loss_units: int
    encoded_record_bytes: int
    persistent_encoded_bytes: int
    decoded_full_state_bytes: int
    decoded_code_transient_bytes: int
    decode_transient_group: str
    minimum_bit_width: int
    exact_bf16: bool
    record_names: tuple[str, ...] = ()
    record_sha256s: tuple[tuple[str, str], ...] = ()
    bundle_sha256: str = ""
    tensor_record_count: int = 1
    auxiliary_record_count: int = 0

    @property
    def runtime_packed_bytes(self) -> int:
        """Backward-compatible alias for persistent encoded bytes."""

        return self.persistent_encoded_bytes


@dataclass(frozen=True, slots=True)
class CrevfBudgetPlan:
    """One exact, deterministic, floor-compliant selection."""

    reduction_ppm: int
    dense_reference_bytes: int
    target_checkpoint_bytes: int
    effective_checkpoint_ceiling_bytes: int
    must_beat_physical_bytes: int | None
    shared_checkpoint_bytes: int
    shared_persistent_encoded_bytes: int
    shared_decoded_full_state_bytes: int
    artifact_physical_bytes: int
    checkpoint_physical_bytes: int
    persistent_encoded_bytes: int
    total_persistent_encoded_bytes: int
    decoded_full_state_bytes: int
    total_decoded_full_state_bytes: int
    maximum_tensor_decode_transient_bytes: int
    maximum_layer_decode_transient_bytes: int
    layer_decode_transient_bytes: tuple[tuple[str, int], ...]
    headroom_bytes: int
    effective_headroom_bytes: int
    max_persistent_encoded_bytes: int | None
    persistent_encoded_headroom_bytes: int | None
    max_decoded_full_state_bytes: int | None
    decoded_full_state_headroom_bytes: int | None
    max_layer_transient_bytes: int | None
    layer_transient_headroom_bytes: int | None
    total_loss_units: int
    frontier_state_count: int
    backpointer_node_upper_bound: int
    max_backpointer_nodes: int
    choices: tuple[CrevfPlanChoice, ...]
    _selected_options: tuple[CrevfSiteOption, ...] = field(repr=False)

    def build_artifact(self) -> bytes:
        artifact = encode_crevf_artifact(
            (
                blob
                for option in self._selected_options
                for blob in option.tensor_blobs
            ),
            auxiliary_blobs=(
                blob
                for option in self._selected_options
                for blob in option.auxiliary_blobs
            ),
        )
        if len(artifact) != self.artifact_physical_bytes:
            raise CrevfBudgetError(
                "Actual CREVF artifact bytes diverged from the sealed planner count."
            )
        if (
            len(artifact) + self.shared_checkpoint_bytes
            != self.checkpoint_physical_bytes
        ):
            raise CrevfBudgetError(
                "Actual CREVF checkpoint bytes diverged from the sealed planner count."
            )
        return artifact

    @property
    def selected_option_ids(self) -> dict[str, str]:
        return {choice.site_id: choice.option_id for choice in self.choices}

    @property
    def runtime_packed_bytes(self) -> int:
        """Backward-compatible alias for artifact persistent encoded bytes."""

        return self.persistent_encoded_bytes

    @property
    def shared_runtime_bytes(self) -> int:
        return self.shared_persistent_encoded_bytes

    @property
    def total_runtime_bytes(self) -> int:
        return self.total_persistent_encoded_bytes

    @property
    def max_runtime_bytes(self) -> int | None:
        return self.max_persistent_encoded_bytes

    @property
    def runtime_headroom_bytes(self) -> int | None:
        return self.persistent_encoded_headroom_bytes


@dataclass(frozen=True, slots=True)
class _FrontierState:
    loss_units: int
    persistent_encoded_bytes: int
    decoded_full_state_bytes: int
    parent: "_FrontierState | None" = field(repr=False)
    option: CrevfSiteOption | None = field(repr=False)

    @property
    def selected_options(self) -> tuple[CrevfSiteOption, ...]:
        output = []
        state: _FrontierState | None = self
        while state is not None and state.option is not None:
            output.append(state.option)
            state = state.parent
        output.reverse()
        return tuple(output)

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(option.option_id for option in self.selected_options)

    @property
    def tie_key(self) -> tuple[Any, ...]:
        return (
            self.loss_units,
            self.persistent_encoded_bytes,
            self.decoded_full_state_bytes,
            self.option_ids,
        )


def _pareto_prune(
    states: dict[int, _FrontierState],
) -> dict[int, _FrontierState]:
    """Remove states dominated by a smaller/equal byte and loss total."""

    output: dict[int, _FrontierState] = {}
    best_loss: int | None = None
    for encoded_bytes in sorted(states):
        state = states[encoded_bytes]
        if best_loss is None or state.loss_units < best_loss:
            output[encoded_bytes] = state
            best_loss = state.loss_units
    return output


def _pareto_prune_with_runtime(
    states: dict[tuple[int, int], _FrontierState],
) -> dict[tuple[int, int], _FrontierState]:
    """Prune states dominated in physical bytes, runtime bytes, and loss."""

    runtime_values = sorted({runtime for _physical, runtime in states})
    runtime_rank = {value: index + 1 for index, value in enumerate(runtime_values)}
    tree: list[int | None] = [None] * (len(runtime_values) + 1)

    def query(index: int) -> int | None:
        result: int | None = None
        while index:
            value = tree[index]
            if value is not None and (result is None or value < result):
                result = value
            index -= index & -index
        return result

    def update(index: int, loss: int) -> None:
        while index < len(tree):
            if tree[index] is None or loss < tree[index]:
                tree[index] = loss
            index += index & -index

    output: dict[tuple[int, int], _FrontierState] = {}
    for key in sorted(states):
        _physical, runtime = key
        state = states[key]
        rank = runtime_rank[runtime]
        dominating_loss = query(rank)
        if dominating_loss is None or dominating_loss > state.loss_units:
            output[key] = state
            update(rank, state.loss_units)
    return output


def plan_crevf_sites(
    sites: tuple[CrevfSite, ...] | list[CrevfSite],
    *,
    dense_reference_bytes: int,
    target_reduction: float,
    shared_checkpoint_bytes: int = 0,
    shared_persistent_encoded_bytes: int = 0,
    shared_decoded_full_state_bytes: int = 0,
    must_beat_physical_bytes: int | None = None,
    max_persistent_encoded_bytes: int | None = None,
    max_decoded_full_state_bytes: int | None = None,
    max_layer_transient_bytes: int | None = None,
    shared_runtime_bytes: int | None = None,
    max_runtime_bytes: int | None = None,
    max_frontier_states: int = 250_000,
    max_provisional_states: int = 500_000,
    max_backpointer_nodes: int = 2_000_000,
) -> CrevfBudgetPlan:
    """Solve the exact multiple-choice byte frontier or fail closed.

    Objective order is:

    1. minimum integer loss;
    2. minimum physical checkpoint bytes;
    3. minimum persistent encoded runtime bytes;
    4. minimum full-decoded diagnostic bytes; and
    5. lexicographically smallest option-id sequence.

    No quantization option may bypass its role precision floor, even with a
    favorable supplied loss.
    """

    _nonnegative_int(dense_reference_bytes, "dense_reference_bytes", positive=True)
    _nonnegative_int(shared_checkpoint_bytes, "shared_checkpoint_bytes")
    _nonnegative_int(
        shared_persistent_encoded_bytes,
        "shared_persistent_encoded_bytes",
    )
    _nonnegative_int(
        shared_decoded_full_state_bytes,
        "shared_decoded_full_state_bytes",
    )
    if shared_runtime_bytes is not None:
        _nonnegative_int(shared_runtime_bytes, "shared_runtime_bytes")
        if (
            shared_persistent_encoded_bytes
            and shared_persistent_encoded_bytes != shared_runtime_bytes
        ):
            raise CrevfError(
                "shared_runtime_bytes conflicts with shared_persistent_encoded_bytes."
            )
        shared_persistent_encoded_bytes = shared_runtime_bytes
    if max_runtime_bytes is not None:
        _nonnegative_int(max_runtime_bytes, "max_runtime_bytes")
        if (
            max_persistent_encoded_bytes is not None
            and max_persistent_encoded_bytes != max_runtime_bytes
        ):
            raise CrevfError(
                "max_runtime_bytes conflicts with max_persistent_encoded_bytes."
            )
        max_persistent_encoded_bytes = max_runtime_bytes
    _nonnegative_int(max_frontier_states, "max_frontier_states", positive=True)
    _nonnegative_int(
        max_provisional_states,
        "max_provisional_states",
        positive=True,
    )
    _nonnegative_int(
        max_backpointer_nodes,
        "max_backpointer_nodes",
        positive=True,
    )
    if must_beat_physical_bytes is not None:
        _nonnegative_int(
            must_beat_physical_bytes,
            "must_beat_physical_bytes",
            positive=True,
        )
    for value, name in (
        (max_persistent_encoded_bytes, "max_persistent_encoded_bytes"),
        (max_decoded_full_state_bytes, "max_decoded_full_state_bytes"),
        (max_layer_transient_bytes, "max_layer_transient_bytes"),
    ):
        if value is not None:
            _nonnegative_int(value, name)
    if max_persistent_encoded_bytes is not None:
        if shared_persistent_encoded_bytes > max_persistent_encoded_bytes:
            raise CrevfBudgetError(
                "Shared persistent encoded bytes already exceed their ceiling."
            )
    if (
        max_decoded_full_state_bytes is not None
        and shared_decoded_full_state_bytes > max_decoded_full_state_bytes
    ):
        raise CrevfBudgetError(
            "Shared decoded full-state bytes already exceed their ceiling."
        )
    target = CrevfReductionTarget.from_reduction(target_reduction)
    ceiling = target.checkpoint_ceiling(dense_reference_bytes)
    effective_ceiling = min(
        ceiling,
        (
            must_beat_physical_bytes - 1
            if must_beat_physical_bytes is not None
            else ceiling
        ),
    )
    ordered_sites = tuple(
        sorted(
            tuple(sites),
            key=lambda site: (
                (
                    crevf_decode_transient_group(site.site_id)
                    if max_layer_transient_bytes is not None
                    else ""
                ),
                site.site_id,
            ),
        )
    )
    if not ordered_sites:
        raise CrevfError("At least one CREVF site is required.")
    site_ids = tuple(site.site_id for site in ordered_sites)
    if len(site_ids) != len(set(site_ids)):
        raise CrevfError("CREVF site IDs must be unique.")
    record_owners: dict[str, str] = {}
    for site in ordered_sites:
        for option in site.options:
            for record_name in option.record_names:
                owner = record_owners.setdefault(record_name, site.site_id)
                if owner != site.site_id:
                    raise CrevfError(
                        f"CREVF record {record_name!r} is owned by both "
                        f"{owner!r} and {site.site_id!r}."
                    )
    fixed_bytes = shared_checkpoint_bytes + CREVF_ARTIFACT_FIXED_BYTES
    if fixed_bytes > effective_ceiling:
        raise CrevfBudgetError(
            f"Fixed CREVF bytes require {fixed_bytes}, exceeding the "
            f"{effective_ceiling}-byte effective ceiling."
        )
    site_budget = effective_ceiling - fixed_bytes
    site_runtime_budget = (
        None
        if max_persistent_encoded_bytes is None
        else max_persistent_encoded_bytes - shared_persistent_encoded_bytes
    )
    site_decoded_budget = (
        None
        if max_decoded_full_state_bytes is None
        else max_decoded_full_state_bytes - shared_decoded_full_state_bytes
    )
    eligible_by_site = tuple(site.eligible_options for site in ordered_sites)
    backpointer_node_upper_bound = 1

    def track_backpointer_node() -> None:
        nonlocal backpointer_node_upper_bound
        backpointer_node_upper_bound += 1
        if backpointer_node_upper_bound > max_backpointer_nodes:
            raise CrevfBudgetError(
                "Exact CREVF cumulative backpointer-node upper bound exceeded "
                "max_backpointer_nodes; refusing an unsafe allocation."
            )

    minimum_required = sum(
        min(option.encoded_record_bytes for option in options)
        for options in eligible_by_site
    )
    if minimum_required > site_budget:
        raise CrevfBudgetError(
            f"Hard precision floors require {minimum_required + fixed_bytes} bytes, "
            f"exceeding the exact {effective_ceiling}-byte effective ceiling."
        )
    if site_decoded_budget is not None or max_layer_transient_bytes is not None:
        grouped: list[
            tuple[
                str,
                list[tuple[CrevfSite, tuple[CrevfSiteOption, ...]]],
            ]
        ] = []
        for site, eligible in zip(
            ordered_sites,
            eligible_by_site,
            strict=True,
        ):
            group = crevf_decode_transient_group(site.site_id)
            if not grouped or grouped[-1][0] != group:
                grouped.append((group, []))
            grouped[-1][1].append((site, eligible))
        multi_frontier: dict[tuple[int, int, int], _FrontierState] = {
            (0, 0, 0): _FrontierState(
                loss_units=0,
                persistent_encoded_bytes=0,
                decoded_full_state_bytes=0,
                parent=None,
                option=None,
            )
        }
        for group, grouped_sites in grouped:
            layer_frontier: dict[
                tuple[int, int, int, int],
                _FrontierState,
            ] = {
                (physical, persistent, decoded, 0): state
                for (physical, persistent, decoded), state in multi_frontier.items()
            }
            for site, eligible in grouped_sites:
                next_layer_states: dict[
                    tuple[int, int, int, int],
                    _FrontierState,
                ] = {}
                for (
                    used_bytes,
                    used_persistent,
                    used_decoded,
                    used_layer_transient,
                ), state in layer_frontier.items():
                    for option in eligible:
                        candidate_bytes = used_bytes + option.encoded_record_bytes
                        candidate_persistent = (
                            used_persistent + option.persistent_encoded_bytes
                        )
                        candidate_decoded = (
                            used_decoded + option.decoded_full_state_bytes
                        )
                        candidate_layer_transient = (
                            used_layer_transient + option.decoded_code_transient_bytes
                        )
                        if (
                            candidate_bytes > site_budget
                            or (
                                site_runtime_budget is not None
                                and candidate_persistent > site_runtime_budget
                            )
                            or (
                                site_decoded_budget is not None
                                and candidate_decoded > site_decoded_budget
                            )
                            or (
                                max_layer_transient_bytes is not None
                                and candidate_layer_transient
                                > max_layer_transient_bytes
                            )
                        ):
                            continue
                        candidate = _FrontierState(
                            loss_units=state.loss_units + option.loss_units,
                            persistent_encoded_bytes=candidate_persistent,
                            decoded_full_state_bytes=candidate_decoded,
                            parent=state,
                            option=option,
                        )
                        key = (
                            candidate_bytes,
                            candidate_persistent,
                            candidate_decoded,
                            candidate_layer_transient,
                        )
                        current = next_layer_states.get(key)
                        if current is None or candidate.tie_key < current.tie_key:
                            next_layer_states[key] = candidate
                            track_backpointer_node()
                            if len(next_layer_states) > max_provisional_states:
                                raise CrevfBudgetError(
                                    "Exact CREVF multi-budget provisional "
                                    "frontier exceeded max_provisional_states "
                                    f"inside layer group {group!r}; refusing an "
                                    "unsafe allocation."
                                )
                if not next_layer_states:
                    raise CrevfBudgetError(
                        "No four-budget floor-compliant CREVF plan remains "
                        f"after site {site.site_id!r}."
                    )
                # Safe partial pruning: within an identical decoded/active
                # resource bucket, physical/persistent/loss dominance is exact.
                buckets: dict[
                    tuple[int, int],
                    dict[tuple[int, int], _FrontierState],
                ] = {}
                for (
                    physical,
                    persistent,
                    decoded,
                    active,
                ), state in next_layer_states.items():
                    buckets.setdefault((decoded, active), {})[
                        (physical, persistent)
                    ] = state
                layer_frontier = {}
                for (decoded, active), bucket in buckets.items():
                    for (
                        physical,
                        persistent,
                    ), state in _pareto_prune_with_runtime(bucket).items():
                        layer_frontier[(physical, persistent, decoded, active)] = state
                if len(layer_frontier) > max_frontier_states:
                    raise CrevfBudgetError(
                        "Exact CREVF multi-budget frontier exceeded "
                        "max_frontier_states; refusing an approximate or "
                        "silently truncated plan."
                    )
            collapsed: dict[
                tuple[int, int, int],
                _FrontierState,
            ] = {}
            for (
                physical,
                persistent,
                decoded,
                _active,
            ), state in layer_frontier.items():
                key = (physical, persistent, decoded)
                current = collapsed.get(key)
                if current is None or state.tie_key < current.tie_key:
                    collapsed[key] = state
            decoded_buckets: dict[
                int,
                dict[tuple[int, int], _FrontierState],
            ] = {}
            for (
                physical,
                persistent,
                decoded,
            ), state in collapsed.items():
                decoded_buckets.setdefault(decoded, {})[(physical, persistent)] = state
            multi_frontier = {}
            for decoded, bucket in decoded_buckets.items():
                for (
                    physical,
                    persistent,
                ), state in _pareto_prune_with_runtime(bucket).items():
                    multi_frontier[(physical, persistent, decoded)] = state
            if len(multi_frontier) > max_frontier_states:
                raise CrevfBudgetError(
                    "Exact CREVF collapsed multi-budget frontier exceeded "
                    "max_frontier_states; refusing an approximate or silently "
                    "truncated plan."
                )
        (
            (
                selected_bytes,
                _selected_persistent,
                _selected_decoded,
            ),
            selected,
        ) = min(
            multi_frontier.items(),
            key=lambda item: (
                item[1].loss_units,
                item[0][0],
                item[0][1],
                item[0][2],
                item[1].option_ids,
            ),
        )
        frontier_count = len(multi_frontier)
    elif site_runtime_budget is None:
        frontier: dict[int, _FrontierState] = {
            0: _FrontierState(
                loss_units=0,
                persistent_encoded_bytes=0,
                decoded_full_state_bytes=0,
                parent=None,
                option=None,
            )
        }
        for site, eligible in zip(ordered_sites, eligible_by_site, strict=True):
            next_states: dict[int, _FrontierState] = {}
            for used_bytes, state in frontier.items():
                for option in eligible:
                    candidate_bytes = used_bytes + option.encoded_record_bytes
                    if candidate_bytes > site_budget:
                        continue
                    candidate = _FrontierState(
                        loss_units=state.loss_units + option.loss_units,
                        persistent_encoded_bytes=(
                            state.persistent_encoded_bytes
                            + option.persistent_encoded_bytes
                        ),
                        decoded_full_state_bytes=(
                            state.decoded_full_state_bytes
                            + option.decoded_full_state_bytes
                        ),
                        parent=state,
                        option=option,
                    )
                    current = next_states.get(candidate_bytes)
                    if current is None or candidate.tie_key < current.tie_key:
                        next_states[candidate_bytes] = candidate
                        track_backpointer_node()
                        if len(next_states) > max_provisional_states:
                            raise CrevfBudgetError(
                                "Exact CREVF provisional frontier exceeded "
                                "max_provisional_states before pruning; refusing "
                                "an unsafe allocation."
                            )
            if not next_states:
                raise CrevfBudgetError(
                    f"No floor-compliant CREVF plan remains after site {site.site_id!r}."
                )
            frontier = _pareto_prune(next_states)
            if len(frontier) > max_frontier_states:
                raise CrevfBudgetError(
                    "Exact CREVF frontier exceeded max_frontier_states; refusing an "
                    "approximate or silently truncated plan."
                )
        selected_bytes, selected = min(
            frontier.items(),
            key=lambda item: (
                item[1].loss_units,
                item[0],
                item[1].persistent_encoded_bytes,
                item[1].decoded_full_state_bytes,
                item[1].option_ids,
            ),
        )
        frontier_count = len(frontier)
    else:
        runtime_frontier: dict[tuple[int, int], _FrontierState] = {
            (0, 0): _FrontierState(
                loss_units=0,
                persistent_encoded_bytes=0,
                decoded_full_state_bytes=0,
                parent=None,
                option=None,
            )
        }
        for site, eligible in zip(ordered_sites, eligible_by_site, strict=True):
            next_runtime_states: dict[tuple[int, int], _FrontierState] = {}
            for (used_bytes, used_runtime), state in runtime_frontier.items():
                for option in eligible:
                    candidate_bytes = used_bytes + option.encoded_record_bytes
                    candidate_runtime = used_runtime + option.persistent_encoded_bytes
                    if (
                        candidate_bytes > site_budget
                        or candidate_runtime > site_runtime_budget
                    ):
                        continue
                    candidate = _FrontierState(
                        loss_units=state.loss_units + option.loss_units,
                        persistent_encoded_bytes=candidate_runtime,
                        decoded_full_state_bytes=(
                            state.decoded_full_state_bytes
                            + option.decoded_full_state_bytes
                        ),
                        parent=state,
                        option=option,
                    )
                    key = (candidate_bytes, candidate_runtime)
                    current = next_runtime_states.get(key)
                    if current is None or (
                        candidate.loss_units,
                        candidate.option_ids,
                    ) < (
                        current.loss_units,
                        current.option_ids,
                    ):
                        next_runtime_states[key] = candidate
                        track_backpointer_node()
                        if len(next_runtime_states) > max_provisional_states:
                            raise CrevfBudgetError(
                                "Exact CREVF provisional physical/runtime frontier "
                                "exceeded max_provisional_states before pruning; "
                                "refusing an unsafe allocation."
                            )
            if not next_runtime_states:
                raise CrevfBudgetError(
                    f"No physical/runtime-floor-compliant CREVF plan remains "
                    f"after site {site.site_id!r}."
                )
            runtime_frontier = _pareto_prune_with_runtime(next_runtime_states)
            if len(runtime_frontier) > max_frontier_states:
                raise CrevfBudgetError(
                    "Exact CREVF physical/runtime frontier exceeded "
                    "max_frontier_states; refusing an approximate or silently "
                    "truncated plan."
                )
        (selected_bytes, _selected_runtime), selected = min(
            runtime_frontier.items(),
            key=lambda item: (
                item[1].loss_units,
                item[0][0],
                item[0][1],
                item[1].decoded_full_state_bytes,
                item[1].option_ids,
            ),
        )
        frontier_count = len(runtime_frontier)
    artifact_bytes = CREVF_ARTIFACT_FIXED_BYTES + selected_bytes
    checkpoint_bytes = shared_checkpoint_bytes + artifact_bytes
    if checkpoint_bytes > effective_ceiling:
        raise CrevfBudgetError("Internal CREVF plan exceeded its exact ceiling.")
    selected_options = selected.selected_options
    selected_persistent_encoded_bytes = sum(
        option.persistent_encoded_bytes for option in selected_options
    )
    selected_decoded_full_state_bytes = sum(
        option.decoded_full_state_bytes for option in selected_options
    )
    total_decoded_full_state_bytes = (
        selected_decoded_full_state_bytes + shared_decoded_full_state_bytes
    )
    if (
        max_decoded_full_state_bytes is not None
        and total_decoded_full_state_bytes > max_decoded_full_state_bytes
    ):
        raise CrevfBudgetError(
            "The exact physical/persistent optimum requires "
            f"{total_decoded_full_state_bytes} decoded full-state diagnostic "
            "bytes, exceeding the independent "
            f"{max_decoded_full_state_bytes}-byte ceiling."
        )
    layer_transient: dict[str, int] = {}
    for option in selected_options:
        if option.decoded_code_transient_bytes == 0:
            continue
        group = option.decode_transient_group
        layer_transient[group] = (
            layer_transient.get(group, 0) + option.decoded_code_transient_bytes
        )
    layer_decode_transient_bytes = tuple(sorted(layer_transient.items()))
    maximum_layer_decode_transient_bytes = max(
        (value for _group, value in layer_decode_transient_bytes),
        default=0,
    )
    if (
        max_layer_transient_bytes is not None
        and maximum_layer_decode_transient_bytes > max_layer_transient_bytes
    ):
        raise CrevfBudgetError(
            "The exact physical/persistent optimum requires "
            f"{maximum_layer_decode_transient_bytes} decoded code bytes in one "
            f"layer, exceeding the independent {max_layer_transient_bytes}-byte "
            "layer ceiling."
        )
    choices = tuple(
        CrevfPlanChoice(
            site_id=site.site_id,
            role=site.role,
            semantic_kind=site.semantic_kind,
            option_id=option.option_id,
            loss_units=option.loss_units,
            encoded_record_bytes=option.encoded_record_bytes,
            persistent_encoded_bytes=option.persistent_encoded_bytes,
            decoded_full_state_bytes=option.decoded_full_state_bytes,
            decoded_code_transient_bytes=option.decoded_code_transient_bytes,
            decode_transient_group=option.decode_transient_group,
            minimum_bit_width=option.minimum_bit_width,
            exact_bf16=option.exact_bf16,
            record_names=option.record_names,
            record_sha256s=option.record_sha256s,
            bundle_sha256=option.bundle_sha256,
            tensor_record_count=len(option.tensor_metadata),
            auxiliary_record_count=len(option.auxiliary_metadata),
        )
        for site, option in zip(ordered_sites, selected_options, strict=True)
    )
    return CrevfBudgetPlan(
        reduction_ppm=target.reduction_ppm,
        dense_reference_bytes=dense_reference_bytes,
        target_checkpoint_bytes=ceiling,
        effective_checkpoint_ceiling_bytes=effective_ceiling,
        must_beat_physical_bytes=must_beat_physical_bytes,
        shared_checkpoint_bytes=shared_checkpoint_bytes,
        shared_persistent_encoded_bytes=shared_persistent_encoded_bytes,
        shared_decoded_full_state_bytes=shared_decoded_full_state_bytes,
        artifact_physical_bytes=artifact_bytes,
        checkpoint_physical_bytes=checkpoint_bytes,
        persistent_encoded_bytes=selected_persistent_encoded_bytes,
        total_persistent_encoded_bytes=(
            selected_persistent_encoded_bytes + shared_persistent_encoded_bytes
        ),
        decoded_full_state_bytes=selected_decoded_full_state_bytes,
        total_decoded_full_state_bytes=total_decoded_full_state_bytes,
        maximum_tensor_decode_transient_bytes=max(
            (
                option.maximum_tensor_decode_transient_bytes
                for option in selected_options
            ),
            default=0,
        ),
        maximum_layer_decode_transient_bytes=(maximum_layer_decode_transient_bytes),
        layer_decode_transient_bytes=layer_decode_transient_bytes,
        headroom_bytes=ceiling - checkpoint_bytes,
        effective_headroom_bytes=effective_ceiling - checkpoint_bytes,
        max_persistent_encoded_bytes=max_persistent_encoded_bytes,
        persistent_encoded_headroom_bytes=(
            None
            if max_persistent_encoded_bytes is None
            else max_persistent_encoded_bytes
            - selected_persistent_encoded_bytes
            - shared_persistent_encoded_bytes
        ),
        max_decoded_full_state_bytes=max_decoded_full_state_bytes,
        decoded_full_state_headroom_bytes=(
            None
            if max_decoded_full_state_bytes is None
            else max_decoded_full_state_bytes - total_decoded_full_state_bytes
        ),
        max_layer_transient_bytes=max_layer_transient_bytes,
        layer_transient_headroom_bytes=(
            None
            if max_layer_transient_bytes is None
            else max_layer_transient_bytes - maximum_layer_decode_transient_bytes
        ),
        total_loss_units=selected.loss_units,
        frontier_state_count=frontier_count,
        backpointer_node_upper_bound=backpointer_node_upper_bound,
        max_backpointer_nodes=max_backpointer_nodes,
        choices=choices,
        _selected_options=selected_options,
    )


__all__ = [
    "CrevfBudgetPlan",
    "CrevfPlanChoice",
    "CrevfSite",
    "CrevfSiteOption",
    "plan_crevf_sites",
    "role_minimum_integer_width",
    "role_requires_exact_bf16",
    "semantic_minimum_integer_width",
    "tensor_identity_minimum_width",
]
