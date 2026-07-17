"""Transactional tied-vocabulary interface folding for local LFM2 runtimes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from neural_forge.hf_factorized_vocab import FactorizedVocabularyPair
from neural_forge.hf_sparse_residual_vocab import SparseResidualVocabularyPair


class Lfm2VocabularyInterfaceFoldError(RuntimeError):
    """A target, installation, preservation audit, or rollback failed closed."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2VocabularyInterfaceFoldError(
            f"{name} must be a positive integer."
        )
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2VocabularyInterfaceFoldError(
            f"{name} must be in the open interval (0, 1)."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2VocabularyInterfaceFoldError(
            f"{name} must be in the open interval (0, 1)."
        ) from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise Lfm2VocabularyInterfaceFoldError(
            f"{name} must be in the open interval (0, 1)."
        )
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def exact_model_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count logical references and unique Parameter objects exactly."""

    named = tuple(model.named_parameters(remove_duplicate=False))
    unique = {id(parameter): parameter for _, parameter in named}
    return {
        "logical": sum(int(parameter.numel()) for _, parameter in named),
        "unique": sum(int(parameter.numel()) for parameter in unique.values()),
        "logical_parameter_objects": len(named),
        "unique_parameter_objects": len(unique),
    }


@dataclass(frozen=True, slots=True)
class TiedVocabularyTarget:
    """An exact rank and physical count at or below a requested total target."""

    original_parent_unique_parameters: int
    source_unique_parameters: int
    dense_interface_unique_parameters: int
    per_rank_unique_parameters: int
    requested_total_compression_fraction: float
    requested_target_unique_parameters: int
    rank: int
    physical_target_unique_parameters: int
    physical_total_compression_fraction: float
    budget_headroom: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_parent_unique_parameters": (
                self.original_parent_unique_parameters
            ),
            "source_unique_parameters": self.source_unique_parameters,
            "dense_interface_unique_parameters": (
                self.dense_interface_unique_parameters
            ),
            "per_rank_unique_parameters": self.per_rank_unique_parameters,
            "requested_total_compression_fraction": (
                self.requested_total_compression_fraction
            ),
            "requested_target_unique_parameters": (
                self.requested_target_unique_parameters
            ),
            "rank": self.rank,
            "physical_target_unique_parameters": (
                self.physical_target_unique_parameters
            ),
            "physical_total_compression_fraction": (
                self.physical_total_compression_fraction
            ),
            "budget_headroom": self.budget_headroom,
        }


def solve_tied_vocabulary_target(
    *,
    original_parent_unique_parameters: int,
    source_unique_parameters: int,
    vocab_size: int,
    hidden_size: int,
    total_compression_fraction: float,
    head_bias_parameters: int = 0,
) -> TiedVocabularyTarget:
    """Choose the largest positive tied rank whose total count fits the target."""

    parent = _positive_int(
        original_parent_unique_parameters,
        "original_parent_unique_parameters",
    )
    source = _positive_int(source_unique_parameters, "source_unique_parameters")
    vocab = _positive_int(vocab_size, "vocab_size")
    hidden = _positive_int(hidden_size, "hidden_size")
    if (
        isinstance(head_bias_parameters, bool)
        or not isinstance(head_bias_parameters, int)
        or head_bias_parameters < 0
    ):
        raise Lfm2VocabularyInterfaceFoldError(
            "head_bias_parameters must be a non-negative integer."
        )
    fraction = _fraction(
        total_compression_fraction,
        "total_compression_fraction",
    )
    dense_interface = vocab * hidden + head_bias_parameters
    per_rank = vocab + hidden
    requested_target = math.floor(parent * (1.0 - fraction))
    fixed = source - dense_interface + head_bias_parameters
    rank = min(hidden, (requested_target - fixed) // per_rank)
    if rank < 1:
        raise Lfm2VocabularyInterfaceFoldError(
            "The requested total target cannot retain a positive vocabulary rank."
        )
    physical = fixed + rank * per_rank
    if physical > requested_target or physical >= source:
        raise Lfm2VocabularyInterfaceFoldError(
            "The solved vocabulary target is not a real compression at/below budget."
        )
    return TiedVocabularyTarget(
        original_parent_unique_parameters=parent,
        source_unique_parameters=source,
        dense_interface_unique_parameters=dense_interface,
        per_rank_unique_parameters=per_rank,
        requested_total_compression_fraction=fraction,
        requested_target_unique_parameters=requested_target,
        rank=rank,
        physical_target_unique_parameters=physical,
        physical_total_compression_fraction=(parent - physical) / parent,
        budget_headroom=requested_target - physical,
    )


def _module_names(model: nn.Module, target: nn.Module) -> tuple[str, ...]:
    names = tuple(
        name
        for name, module in model.named_modules(remove_duplicate=False)
        if module is target
    )
    if not names or any(not name for name in names):
        raise Lfm2VocabularyInterfaceFoldError(
            "Vocabulary module is not installed at a named child path."
        )
    return names


def _under_roots(name: str, roots: tuple[str, ...]) -> bool:
    return any(name == root or name.startswith(f"{root}.") for root in roots)


@dataclass(frozen=True, slots=True)
class NonVocabularySnapshot:
    """Object-identity and byte hashes for every non-interface model surface."""

    interface_roots: tuple[str, ...]
    modules: Mapping[str, tuple[int, str]]
    parameters: Mapping[str, tuple[int, tuple[int, ...], str, str]]
    buffers: Mapping[str, tuple[int, tuple[int, ...], str, str]]
    snapshot_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "interface_roots": list(self.interface_roots),
            "module_count": len(self.modules),
            "parameter_reference_count": len(self.parameters),
            "buffer_reference_count": len(self.buffers),
            "snapshot_sha256": self.snapshot_sha256,
        }


def capture_non_vocabulary_snapshot(
    model: nn.Module,
    embedding: nn.Module,
    lm_head: nn.Module,
) -> NonVocabularySnapshot:
    """Seal all modules, parameters, and buffers outside the two vocab roots."""

    if not isinstance(model, nn.Module):
        raise Lfm2VocabularyInterfaceFoldError("model must be an nn.Module.")
    roots = tuple(
        dict.fromkeys(_module_names(model, embedding) + _module_names(model, lm_head))
    )
    modules = {
        name: (
            id(module),
            f"{type(module).__module__}.{type(module).__qualname__}",
        )
        for name, module in model.named_modules(remove_duplicate=False)
        if name and not _under_roots(name, roots)
    }
    parameters = {
        name: (
            id(parameter),
            tuple(int(value) for value in parameter.shape),
            str(parameter.dtype),
            _tensor_sha256(parameter),
        )
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if not _under_roots(name, roots)
    }
    buffers = {
        name: (
            id(buffer),
            tuple(int(value) for value in buffer.shape),
            str(buffer.dtype),
            _tensor_sha256(buffer),
        )
        for name, buffer in model.named_buffers(remove_duplicate=False)
        if not _under_roots(name, roots)
    }
    body = {
        "interface_roots": list(roots),
        "modules": {
            name: [identity, kind] for name, (identity, kind) in sorted(modules.items())
        },
        "parameters": {
            name: [identity, list(shape), dtype, digest]
            for name, (identity, shape, dtype, digest) in sorted(parameters.items())
        },
        "buffers": {
            name: [identity, list(shape), dtype, digest]
            for name, (identity, shape, dtype, digest) in sorted(buffers.items())
        },
    }
    return NonVocabularySnapshot(
        interface_roots=roots,
        modules=modules,
        parameters=parameters,
        buffers=buffers,
        snapshot_sha256=_canonical_sha256(body),
    )


def audit_non_vocabulary_snapshot(
    model: nn.Module,
    snapshot: NonVocabularySnapshot,
) -> dict[str, Any]:
    """Require every sealed non-vocabulary object and tensor byte to be exact."""

    if not isinstance(snapshot, NonVocabularySnapshot):
        raise Lfm2VocabularyInterfaceFoldError(
            "snapshot must be a NonVocabularySnapshot."
        )
    modules = {
        name: (
            id(module),
            f"{type(module).__module__}.{type(module).__qualname__}",
        )
        for name, module in model.named_modules(remove_duplicate=False)
        if name and not _under_roots(name, snapshot.interface_roots)
    }
    parameters = {
        name: (
            id(parameter),
            tuple(int(value) for value in parameter.shape),
            str(parameter.dtype),
            _tensor_sha256(parameter),
        )
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if not _under_roots(name, snapshot.interface_roots)
    }
    buffers = {
        name: (
            id(buffer),
            tuple(int(value) for value in buffer.shape),
            str(buffer.dtype),
            _tensor_sha256(buffer),
        )
        for name, buffer in model.named_buffers(remove_duplicate=False)
        if not _under_roots(name, snapshot.interface_roots)
    }
    exact = bool(
        modules == dict(snapshot.modules)
        and parameters == dict(snapshot.parameters)
        and buffers == dict(snapshot.buffers)
    )
    if not exact:
        raise Lfm2VocabularyInterfaceFoldError(
            "A non-vocabulary module, Parameter, buffer object, or tensor byte changed."
        )
    return {
        **snapshot.as_dict(),
        "module_objects_exact": True,
        "parameter_objects_and_bytes_exact": True,
        "buffer_objects_and_bytes_exact": True,
        "attention_recurrent_and_ffn_objects_exact": True,
        "exact": True,
    }


@dataclass(slots=True)
class InstalledLfm2VocabularyInterface:
    """Live reversible installation of one physically shared factor pair."""

    model: nn.Module
    original_embedding: nn.Module
    original_lm_head: nn.Module
    pair: FactorizedVocabularyPair | SparseResidualVocabularyPair
    source_parameter_counts: Mapping[str, int]
    candidate_parameter_counts: Mapping[str, int]
    non_vocabulary_snapshot: NonVocabularySnapshot
    installation_audit: Mapping[str, Any]
    active: bool = True

    def rollback(self) -> bool:
        if not self.active:
            return False
        try:
            self.model.set_input_embeddings(self.original_embedding)
            self.model.set_output_embeddings(self.original_lm_head)
        except BaseException as exc:
            raise Lfm2VocabularyInterfaceFoldError(
                "Could not restore the original vocabulary modules."
            ) from exc
        if (
            self.model.get_input_embeddings() is not self.original_embedding
            or self.model.get_output_embeddings() is not self.original_lm_head
            or exact_model_parameter_counts(self.model)
            != dict(self.source_parameter_counts)
        ):
            raise Lfm2VocabularyInterfaceFoldError(
                "Vocabulary rollback did not restore the exact source surface."
            )
        audit_non_vocabulary_snapshot(self.model, self.non_vocabulary_snapshot)
        self.active = False
        return True


def install_lfm2_tied_vocabulary_pair(
    model: nn.Module,
    pair: FactorizedVocabularyPair | SparseResidualVocabularyPair,
) -> InstalledLfm2VocabularyInterface:
    """Install one tied factor pair through the model's public module setters."""

    if not isinstance(model, nn.Module):
        raise Lfm2VocabularyInterfaceFoldError("model must be an nn.Module.")
    if not isinstance(
        pair,
        (FactorizedVocabularyPair, SparseResidualVocabularyPair),
    ) or (
        isinstance(pair, FactorizedVocabularyPair) and not pair.tied
    ):
        raise Lfm2VocabularyInterfaceFoldError(
            "pair must be a truly tied factorized vocabulary pair."
        )
    for name in (
        "get_input_embeddings",
        "get_output_embeddings",
        "set_input_embeddings",
        "set_output_embeddings",
    ):
        if not callable(getattr(model, name, None)):
            raise Lfm2VocabularyInterfaceFoldError(
                f"model does not implement {name}()."
            )
    original_embedding = model.get_input_embeddings()
    original_head = model.get_output_embeddings()
    if not isinstance(original_embedding, nn.Embedding) or not isinstance(
        original_head, nn.Linear
    ):
        raise Lfm2VocabularyInterfaceFoldError(
            "The source LFM2 vocabulary modules must be dense Embedding/Linear."
        )
    if original_embedding.weight is not original_head.weight:
        raise Lfm2VocabularyInterfaceFoldError(
            "The source LFM2 embedding and LM head are not the same Parameter."
        )
    if (
        original_embedding.num_embeddings != pair.embedding.num_embeddings
        or original_embedding.embedding_dim != pair.embedding.embedding_dim
        or original_head.out_features != pair.lm_head.out_features
        or original_head.in_features != pair.lm_head.in_features
    ):
        raise Lfm2VocabularyInterfaceFoldError(
            "Factorized and dense vocabulary interface dimensions differ."
        )
    source_counts = exact_model_parameter_counts(model)
    snapshot = capture_non_vocabulary_snapshot(
        model,
        original_embedding,
        original_head,
    )
    original_interface = sum(
        int(parameter.numel())
        for parameter in {
            id(parameter): parameter
            for module in (original_embedding, original_head)
            for parameter in module.parameters()
        }.values()
    )
    expected_candidate = (
        source_counts["unique"] - original_interface + pair.unique_parameter_count
    )
    installed_input = False
    try:
        model.set_input_embeddings(pair.embedding)
        installed_input = True
        model.set_output_embeddings(pair.lm_head)
        candidate_counts = exact_model_parameter_counts(model)
        if isinstance(pair, SparseResidualVocabularyPair):
            shared_physical_state = bool(
                pair.embedding.core is pair.lm_head.core
                and pair.embedding.core.vocab_factor
                is pair.lm_head.core.vocab_factor
                and pair.embedding.core.hidden_factor
                is pair.lm_head.core.hidden_factor
                and pair.embedding.core.residual_rows
                is pair.lm_head.core.residual_rows
                and pair.embedding.core.selected_token_ids
                is pair.lm_head.core.selected_token_ids
            )
        else:
            shared_physical_state = bool(
                pair.embedding.vocab_factor is pair.lm_head.vocab_factor
                and pair.embedding.hidden_factor is pair.lm_head.hidden_factor
            )
        if (
            model.get_input_embeddings() is not pair.embedding
            or model.get_output_embeddings() is not pair.lm_head
            or not shared_physical_state
            or candidate_counts["unique"] != expected_candidate
        ):
            raise Lfm2VocabularyInterfaceFoldError(
                "Installed tied vocabulary factors failed identity or accounting."
            )
        preservation = audit_non_vocabulary_snapshot(model, snapshot)
    except BaseException as exc:
        try:
            if installed_input:
                model.set_input_embeddings(original_embedding)
            model.set_output_embeddings(original_head)
        except BaseException as rollback_exc:
            raise Lfm2VocabularyInterfaceFoldError(
                "Vocabulary installation and emergency rollback both failed."
            ) from rollback_exc
        raise Lfm2VocabularyInterfaceFoldError(
            "Could not install the tied vocabulary factor pair."
        ) from exc
    audit = {
        "source_parameter_counts": source_counts,
        "candidate_parameter_counts": candidate_counts,
        "dense_interface_unique_parameters": original_interface,
        "factor_interface_unique_parameters": pair.unique_parameter_count,
        "physical_parameters_removed": source_counts["unique"]
        - candidate_counts["unique"],
        "embedding_and_head_share_exact_factor_parameter_objects": True,
        "sparse_residual_runtime": isinstance(
            pair,
            SparseResidualVocabularyPair,
        ),
        "dense_vocabulary_weight_retained": False,
        "factor_parameter_report": pair.parameter_report(),
        "non_vocabulary_preservation": preservation,
        "exact": True,
    }
    return InstalledLfm2VocabularyInterface(
        model=model,
        original_embedding=original_embedding,
        original_lm_head=original_head,
        pair=pair,
        source_parameter_counts=source_counts,
        candidate_parameter_counts=candidate_counts,
        non_vocabulary_snapshot=snapshot,
        installation_audit=audit,
    )


def install_lfm2_sparse_residual_vocabulary_pair(
    model: nn.Module,
    pair: SparseResidualVocabularyPair,
) -> InstalledLfm2VocabularyInterface:
    """Install one shared low-rank-plus-sparse-residual vocabulary core."""

    if not isinstance(pair, SparseResidualVocabularyPair):
        raise Lfm2VocabularyInterfaceFoldError(
            "pair must be a SparseResidualVocabularyPair."
        )
    return install_lfm2_tied_vocabulary_pair(model, pair)


__all__ = [
    "InstalledLfm2VocabularyInterface",
    "Lfm2VocabularyInterfaceFoldError",
    "NonVocabularySnapshot",
    "TiedVocabularyTarget",
    "audit_non_vocabulary_snapshot",
    "capture_non_vocabulary_snapshot",
    "exact_model_parameter_counts",
    "install_lfm2_tied_vocabulary_pair",
    "install_lfm2_sparse_residual_vocabulary_pair",
    "solve_tied_vocabulary_target",
]
