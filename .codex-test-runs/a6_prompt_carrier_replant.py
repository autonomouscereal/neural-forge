from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


RUNTIME = Path("/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2")
sys.path.insert(0, str(RUNTIME / "src"))

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_model_runtime import (  # noqa: E402
    _donor_prompt_ids,
    run_lfm2_semantic_step_model_review,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    bind_lfm2_separate_sparse_semantic_code_runtime,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (  # noqa: E402
    load_lfm2_sparse_semantic_codebook,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)


PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
CHECKPOINT = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_bridge_only_control_20260723/"
    "89cd45b0/same-body-v4096-p40-gpu1-a0001/checkpoint"
)
DONOR_TOKENIZER = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/"
    "eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
)
TARGET_TOKENIZER = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/"
    "v4096_modelcap_v2"
)
BRIDGE_SHA = "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384"
CODEBOOK = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/"
    "lfm2-sparse-semantic-codebook/"
    "51b42fbcbbcd3c5b638c6a9d2c50a2596173a5e9181db0845b72d6970e15f3c0/"
    "payload/artifact"
)
CODEBOOK_SHA = "280a75b7971743b85dae79f23ddcde3c2c15f4ab77558045dce0fe2431e5ea2e"
OUTPUT_COEFFICIENTS = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "neural_forge_functional_output_refit_20260730/a5-fixed-a4-support/"
    "output-coefficients-lambda-0p001.safetensors"
)
OUTPUT_COEFFICIENTS_FILE_SHA = (
    "bc4a69eb7d0462bc21431587bb9ec60ead86ae774c4101475ec120897d4c4f55"
)
SPAN = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/"
    "lfm2-semantic-step-global-raw-span/"
    "3e780d07ac1034ee77756f973b483637cee846e17e6d7191d6b20c9533360167/"
    "payload/span-evidence.json"
)
PROMPTS = RUNTIME / "research/lfm2_semantic_whole_response_prompts_v1.json"
ENVELOPE = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/"
    "e739a88/a0002/parent-envelope.json"
)
A4_DIAGNOSTIC = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "neural_forge_sparse_semantic_bridge_20260730/a4-k16-residual-omp/"
    "cached-hidden-diagnostic.json"
)
RUN_DIRECTORY = Path(
    os.environ.get(
        "NF_CARRIER_RUN_DIRECTORY",
        "/home/cereal/model-surgery-lab/artifacts/"
        "neural_forge_carrier_replant_20260730/a6-prompt191-cout-lambda0001",
    )
)
SELECTION_MODE = os.environ.get(
    "NF_CARRIER_SELECTION_MODE", "prompt-diagnostic-v1"
)
EXTERNAL_SELECTION_EVIDENCE = Path(
    os.environ.get(
        "NF_CARRIER_EVIDENCE_PATH",
        str(RUNTIME / ".codex-test-runs/lfm2-natural-language-token-evidence-2025-2026.json"),
    )
)
CARRIER_BUDGET = int(os.environ.get("NF_CARRIER_BUDGET", "356"))
RUNTIME_DTYPE_NAME = os.environ.get("NF_RUNTIME_DTYPE", "bfloat16")
RUNTIME_DEVICE_NAME = os.environ.get("NF_RUNTIME_DEVICE", "cuda:0")
CARRIER_ALLOCATION_ALGORITHM_ABI = "lfm2-semantic-carrier-allocation-v2"
LOCALIZED_OMP_ALGORITHM_ABI = "lfm2-localized-residual-omp-fp32-fp64-v1"
DONOR_COUNT = 64_402
TARGET_COUNT = 4_096
HIDDEN_SIZE = 1_024
OMP_K = 16
OMP_ORTHOGONALITY_TOLERANCE = 64.0 * torch.finfo(torch.float32).eps
COEFFICIENT_RIDGE = 1e-5
MAXIMUM_NEW_TOKENS = 256


def resolve_runtime_dtype(name: str) -> torch.dtype:
    supported = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return supported[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported NF_RUNTIME_DTYPE={name!r}; expected one of "
            f"{sorted(supported)}."
        ) from exc


def emit(stage: str, **fields: object) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prompt_token_counts(
    donor_tokenizer: object,
    prompts: list[dict[str, Any]],
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for prompt in prompts:
        ids = _donor_prompt_ids(donor_tokenizer, prompt)[0].tolist()
        counts.update(int(value) for value in ids)
    return counts


def repository_corpus_token_counts(
    donor_tokenizer: object,
) -> tuple[Counter[int], dict[str, Any]]:
    allowed_suffixes = {
        ".json",
        ".jinja",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    excluded_directories = {
        ".codex-test-runs",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "reports",
    }
    excluded_files = {
        "research/lfm2_semantic_whole_response_prompts_v1.json",
        "research/2026-07-30-lfm2-dual-channel-token-bridge-gap.md",
    }
    maximum_file_bytes = 2 * 1024 * 1024
    inventory: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    for path in sorted(RUNTIME.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        relative = path.relative_to(RUNTIME).as_posix()
        if relative in excluded_files or any(
            part in excluded_directories for part in path.relative_to(RUNTIME).parts
        ):
            continue
        size = path.stat().st_size
        if size > maximum_file_bytes:
            continue
        payload = path.read_bytes()
        inventory.append(
            {
                "path": relative,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        payloads.append((relative, payload))
    policy = {
        "schema": "neural-forge-repository-text-fold-fit-policy-v1",
        "allowed_suffixes": sorted(allowed_suffixes),
        "excluded_directories": sorted(excluded_directories),
        "excluded_files": sorted(excluded_files),
        "maximum_file_bytes": maximum_file_bytes,
        "tokenizer_add_special_tokens": False,
        "review_prompt_tokens_used": False,
        "parent_response_tokens_used": False,
    }
    inventory_sha256 = canonical_sha256(
        {"policy": policy, "files": inventory}
    )
    cache_path = RUN_DIRECTORY / "fold-fit-selection-evidence.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        supplied_sha = cached.get("evidence_sha256")
        body = dict(cached)
        body.pop("evidence_sha256", None)
        if (
            supplied_sha == canonical_sha256(body)
            and cached.get("inventory_sha256") == inventory_sha256
            and cached.get("policy") == policy
        ):
            counts = Counter(
                {
                    int(token_id): int(count)
                    for token_id, count in cached["token_counts"]
                }
            )
            emit(
                "fold-fit-selection-evidence-reused",
                evidence_sha256=supplied_sha,
                file_count=len(inventory),
                token_occurrence_count=sum(counts.values()),
                unique_token_count=len(counts),
            )
            return counts, cached
    emit(
        "fold-fit-selection-evidence-start",
        inventory_sha256=inventory_sha256,
        file_count=len(inventory),
    )
    encode = getattr(donor_tokenizer, "encode")
    counts: Counter[int] = Counter()
    for index, (relative, payload) in enumerate(payloads):
        text = payload.decode("utf-8", errors="replace")
        counts.update(
            int(token_id)
            for token_id in encode(text, add_special_tokens=False)
        )
        if (index + 1) % 50 == 0 or index + 1 == len(payloads):
            emit(
                "fold-fit-selection-evidence-progress",
                files_complete=index + 1,
                files_total=len(payloads),
                latest_file=relative,
                token_occurrence_count=sum(counts.values()),
                unique_token_count=len(counts),
            )
    body = {
        "schema": "neural-forge-lfm2-fold-fit-token-evidence-v1",
        "status": "complete",
        "inventory_sha256": inventory_sha256,
        "policy": policy,
        "files": inventory,
        "token_counts": [[token_id, counts[token_id]] for token_id in sorted(counts)],
        "token_occurrence_count": sum(counts.values()),
        "unique_token_count": len(counts),
    }
    evidence = {**body, "evidence_sha256": canonical_sha256(body)}
    cache_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit(
        "fold-fit-selection-evidence-complete",
        evidence_sha256=evidence["evidence_sha256"],
        token_occurrence_count=sum(counts.values()),
        unique_token_count=len(counts),
    )
    return counts, evidence


def reconstruction_relative_errors(
    *,
    donor_ids: list[int],
    offsets: torch.Tensor,
    target_ids: torch.Tensor,
    coefficients: torch.Tensor,
    parent_table: torch.Tensor,
    target_table: torch.Tensor,
) -> dict[int, float]:
    errors: dict[int, float] = {}
    started = time.monotonic()
    emit("promotion-error-inventory-start", donor_count=len(donor_ids))
    for index, donor_id in enumerate(donor_ids):
        start = int(offsets[donor_id].item())
        stop = int(offsets[donor_id + 1].item())
        support = target_ids[start:stop]
        weights = coefficients[start:stop].to(dtype=torch.float32)
        reconstruction = torch.sum(
            target_table.index_select(0, support).to(dtype=torch.float32)
            * weights.unsqueeze(1),
            dim=0,
        )
        source = parent_table[donor_id].to(dtype=torch.float32)
        denominator = torch.linalg.vector_norm(source)
        if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0:
            raise RuntimeError(f"Donor {donor_id} has invalid embedding norm.")
        relative_l2 = torch.linalg.vector_norm(reconstruction - source) / denominator
        if not bool(torch.isfinite(relative_l2)):
            raise RuntimeError(f"Donor {donor_id} has non-finite reconstruction error.")
        errors[donor_id] = float(relative_l2.item())
        if (index + 1) % 1024 == 0 or index + 1 == len(donor_ids):
            emit(
                "promotion-error-inventory-progress",
                donors_complete=index + 1,
                donors_total=len(donor_ids),
                elapsed_seconds=time.monotonic() - started,
            )
    emit(
        "promotion-error-inventory-complete",
        donor_count=len(errors),
        elapsed_seconds=time.monotonic() - started,
    )
    return errors


def cached_direct_winners(mapping: torch.Tensor) -> set[int]:
    if not A4_DIAGNOSTIC.is_file():
        return set()
    diagnostic = json.loads(A4_DIAGNOSTIC.read_text(encoding="utf-8"))
    direct = frozenset(int(value) for value in mapping.tolist())
    return {
        int(row["parent_winner_donor_id"])
        for row in diagnostic["row_rank_diagnostics"]
        if int(row["parent_winner_donor_id"]) in direct
    }


def select_carrier_reassignments(
    *,
    input_offsets: torch.Tensor,
    input_target_ids: torch.Tensor,
    output_offsets: torch.Tensor,
    output_target_ids: torch.Tensor,
    mapping: torch.Tensor,
    evidence_counts: Counter[int],
    promotion_scores: dict[int, float],
    protected_donor_ids: set[int],
    carrier_budget: int,
) -> tuple[list[int], list[int], dict[str, object]]:
    direct_lookup = torch.zeros(DONOR_COUNT, dtype=torch.bool)
    direct_lookup[mapping] = True
    priority = sorted(
        (
            donor_id
            for donor_id in evidence_counts
            if not bool(direct_lookup[donor_id]) and donor_id in promotion_scores
        ),
        key=lambda donor_id: (
            -promotion_scores[donor_id],
            -evidence_counts[donor_id],
            donor_id,
        ),
    )[:carrier_budget]

    def non_direct_incidence(
        offsets: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        lengths = offsets[1:] - offsets[:-1]
        edge_rows = torch.repeat_interleave(torch.arange(DONOR_COUNT), lengths)
        non_direct_edges = ~direct_lookup.index_select(0, edge_rows)
        return torch.bincount(
            target_ids[non_direct_edges], minlength=TARGET_COUNT
        )

    input_incidence = non_direct_incidence(input_offsets, input_target_ids)
    output_incidence = non_direct_incidence(output_offsets, output_target_ids)
    union_incidence = input_incidence + output_incidence
    eligible = [
        target_id
        for target_id in range(TARGET_COUNT)
        if int(union_incidence[target_id].item()) == 0
        and int(mapping[target_id].item()) not in protected_donor_ids
        and int(mapping[target_id].item()) not in priority
    ]
    eligible.sort(
        key=lambda target_id: (
            evidence_counts[int(mapping[target_id].item())],
            -int(mapping[target_id].item()),
            -target_id,
        )
    )
    if len(eligible) < len(priority):
        raise RuntimeError(
            f"Only {len(eligible)} safe zero-incidence carriers exist for "
            f"{len(priority)} priority donors."
        )
    carriers = eligible[: len(priority)]
    return priority, carriers, {
        "selection_token_occurrence_count": sum(evidence_counts.values()),
        "selection_unique_donor_count": len(evidence_counts),
        "priority_non_direct_donor_count": len(priority),
        "carrier_budget": carrier_budget,
        "zero_union_non_direct_incidence_target_count": int(
            union_incidence.eq(0).sum().item()
        ),
        "eligible_zero_incidence_carrier_count": len(eligible),
        "protected_donor_count": len(protected_donor_ids),
        "selected_target_carriers": carriers,
        "promoted_donor_ids": priority,
        "promoted_donor_evidence_counts": [
            evidence_counts[donor_id] for donor_id in priority
        ],
        "promoted_donor_scores": [promotion_scores[donor_id] for donor_id in priority],
        "evicted_donor_ids": [int(mapping[target_id].item()) for target_id in carriers],
        "evicted_donor_evidence_counts": [
            evidence_counts[int(mapping[target_id].item())] for target_id in carriers
        ],
    }


def localized_residual_omp(
    *,
    source_rows: torch.Tensor,
    dictionary_rows: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    source = source_rows.to(device=device, dtype=torch.float32)
    dictionary = dictionary_rows.to(device=device, dtype=torch.float32)
    source_norms = torch.linalg.vector_norm(source, dim=1)
    dictionary_norms = torch.linalg.vector_norm(dictionary, dim=1)
    if bool((source_norms <= 0).any()) or bool((dictionary_norms <= 0).any()):
        raise RuntimeError("Carrier OMP received a zero-norm semantic row.")
    source_unit = source / source_norms.unsqueeze(1)
    dictionary_unit = dictionary / dictionary_norms.unsqueeze(1)
    row_count = int(source.shape[0])
    selected = torch.empty((row_count, OMP_K), device=device, dtype=torch.int64)
    q = torch.zeros(
        (row_count, OMP_K, HIDDEN_SIZE), device=device, dtype=torch.float32
    )
    residual = source_unit.clone()
    started = time.monotonic()
    with torch.no_grad():
        for step in range(OMP_K):
            correlations = torch.abs(residual @ dictionary_unit.transpose(0, 1))
            if step:
                correlations.scatter_(1, selected[:, :step], -torch.inf)
            chosen = correlations.argmax(dim=1)
            selected[:, step] = chosen
            atom = dictionary_unit.index_select(0, chosen).clone()
            for _ in range(2):
                if step:
                    projection = torch.einsum("nkd,nd->nk", q[:, :step], atom)
                    atom.sub_(torch.einsum("nk,nkd->nd", projection, q[:, :step]))
            atom_norm = torch.linalg.vector_norm(atom, dim=1)
            if bool((atom_norm <= OMP_ORTHOGONALITY_TOLERANCE).any()):
                raise RuntimeError(
                    "Localized carrier OMP encountered a dependent selected atom."
                )
            q[:, step] = atom / atom_norm.unsqueeze(1)
            coordinates = torch.einsum(
                "nkd,nd->nk", q[:, : step + 1], source_unit
            )
            residual = source_unit - torch.einsum(
                "nk,nkd->nd", coordinates, q[:, : step + 1]
            )
            emit(
                "localized-omp-step",
                step=step + 1,
                active_rows=row_count,
                maximum_unit_residual_norm=float(
                    torch.linalg.vector_norm(residual, dim=1).max().item()
                ),
                elapsed_seconds=time.monotonic() - started,
            )
    selected_cpu = selected.to(device="cpu")
    basis = dictionary_rows.index_select(0, selected_cpu.reshape(-1)).reshape(
        row_count, OMP_K, HIDDEN_SIZE
    ).to(dtype=torch.float64)
    source_cpu = source_rows.to(device="cpu", dtype=torch.float64)
    gram = torch.bmm(basis, basis.transpose(1, 2))
    gram.diagonal(dim1=-2, dim2=-1).add_(COEFFICIENT_RIDGE)
    rhs = torch.bmm(basis, source_cpu.unsqueeze(2))
    coefficients = torch.linalg.solve(gram, rhs).squeeze(2)
    order = torch.argsort(selected_cpu, dim=1)
    selected_cpu = selected_cpu.gather(1, order)
    coefficients = coefficients.gather(1, order).to(dtype=torch.float32)
    reconstructed = torch.bmm(
        coefficients.to(dtype=torch.float64).unsqueeze(1),
        dictionary_rows.index_select(0, selected_cpu.reshape(-1))
        .reshape(row_count, OMP_K, HIDDEN_SIZE)
        .to(dtype=torch.float64),
    ).squeeze(1)
    source_norm = torch.linalg.vector_norm(source_cpu, dim=1)
    relative_l2 = torch.linalg.vector_norm(
        reconstructed - source_cpu, dim=1
    ) / source_norm
    cosine = torch.nn.functional.cosine_similarity(
        reconstructed, source_cpu, dim=1
    )
    return selected_cpu, coefficients, {
        "localized_omp_row_count": row_count,
        "localized_omp_k": OMP_K,
        "localized_omp_full_dictionary_search_steps": row_count * OMP_K,
        "localized_omp_selection_device": str(device),
        "localized_omp_selection_dtype": "float32",
        "localized_coefficient_solve_device": "cpu",
        "localized_coefficient_solve_dtype": "float64",
        "localized_coefficient_publication_dtype": "float32",
        "localized_coefficient_ridge": COEFFICIENT_RIDGE,
        "mean_evicted_donor_relative_l2": float(relative_l2.mean().item()),
        "maximum_evicted_donor_relative_l2": float(relative_l2.max().item()),
        "mean_evicted_donor_cosine": float(cosine.mean().item()),
        "minimum_evicted_donor_cosine": float(cosine.min().item()),
        "elapsed_seconds": time.monotonic() - started,
    }


def rewrite_code(
    *,
    base_offsets: torch.Tensor,
    base_target_ids: torch.Tensor,
    base_coefficients: torch.Tensor,
    promoted_to_carrier: dict[int, int],
    evicted_supports: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = [0]
    ids = []
    coefficients = []
    for donor_id in range(DONOR_COUNT):
        if donor_id in promoted_to_carrier:
            row_ids = torch.tensor(
                [promoted_to_carrier[donor_id]], dtype=torch.int64
            )
            row_coefficients = torch.ones(1, dtype=torch.float32)
        elif donor_id in evicted_supports:
            row_ids, row_coefficients = evicted_supports[donor_id]
        else:
            start = int(base_offsets[donor_id].item())
            stop = int(base_offsets[donor_id + 1].item())
            row_ids = base_target_ids[start:stop]
            row_coefficients = base_coefficients[start:stop]
        ids.append(row_ids)
        coefficients.append(row_coefficients)
        offsets.append(offsets[-1] + int(row_ids.numel()))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.cat(ids).to(dtype=torch.int64).contiguous(),
        torch.cat(coefficients).to(dtype=torch.float32).contiguous(),
    )


def compile_or_load_carrier_artifact(
    *,
    donor_tokenizer: object,
    prompts: list[dict[str, Any]],
    parent_table: torch.Tensor,
    original_target_table: torch.Tensor,
    codebook: object,
    output_coefficients: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    artifact_path = RUN_DIRECTORY / "carrier-artifact.safetensors"
    receipt_path = RUN_DIRECTORY / "carrier-artifact.json"
    if SELECTION_MODE == "prompt-diagnostic-v1":
        evidence_counts = prompt_token_counts(donor_tokenizer, prompts)
        selection_evidence = {
            "schema": "neural-forge-review-prompt-token-evidence-v1",
            "selection_mode": SELECTION_MODE,
            "prompts_file_sha256": file_sha256(PROMPTS),
            "review_prompt_tokens_used": True,
            "parent_response_tokens_used": False,
            "token_occurrence_count": sum(evidence_counts.values()),
            "unique_token_count": len(evidence_counts),
        }
        selection_evidence_sha256 = canonical_sha256(selection_evidence)
    elif SELECTION_MODE == "repository-corpus-v1":
        evidence_counts, selection_evidence = repository_corpus_token_counts(
            donor_tokenizer
        )
        selection_evidence_sha256 = str(selection_evidence["evidence_sha256"])
    elif SELECTION_MODE == "natural-language-corpus-v1":
        selection_evidence = json.loads(
            EXTERNAL_SELECTION_EVIDENCE.read_text(encoding="utf-8")
        )
        supplied_sha = selection_evidence.get("evidence_sha256")
        evidence_body = dict(selection_evidence)
        evidence_body.pop("evidence_sha256", None)
        if supplied_sha != canonical_sha256(evidence_body):
            raise RuntimeError("External natural-language evidence seal is invalid.")
        policy = selection_evidence.get("policy", {})
        if policy.get("review_prompt_tokens_used") is not False:
            raise RuntimeError("External selection evidence may not use review prompts.")
        if policy.get("parent_response_tokens_used") is not False:
            raise RuntimeError("External selection evidence may not use parent responses.")
        if int(policy.get("training_steps", -1)) != 0:
            raise RuntimeError("External selection evidence records training operations.")
        evidence_counts = Counter(
            {
                int(token_id): int(count)
                for token_id, count in selection_evidence["token_counts"]
            }
        )
        selection_evidence_sha256 = str(supplied_sha)
        emit(
            "external-fold-fit-selection-evidence-loaded",
            evidence_sha256=supplied_sha,
            file_count=len(selection_evidence["files"]),
            sampled_byte_count=selection_evidence["sampled_byte_count"],
            token_occurrence_count=sum(evidence_counts.values()),
            unique_token_count=len(evidence_counts),
        )
    else:
        raise ValueError(f"Unsupported NF_CARRIER_SELECTION_MODE={SELECTION_MODE!r}.")
    dependency = {
        "carrier_allocation_algorithm_abi": CARRIER_ALLOCATION_ALGORITHM_ABI,
        "localized_omp_algorithm_abi": LOCALIZED_OMP_ALGORITHM_ABI,
        "selection_mode": SELECTION_MODE,
        "selection_evidence_sha256": selection_evidence_sha256,
        "carrier_budget": CARRIER_BUDGET,
        "input_codebook_sha256": CODEBOOK_SHA,
        "output_coefficients_file_sha256": OUTPUT_COEFFICIENTS_FILE_SHA,
        "parent_table_sha256": tensor_sha256(parent_table),
        "original_target_table_sha256": tensor_sha256(original_target_table),
    }
    if artifact_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        supplied_sha = receipt.get("artifact_sha256")
        body = dict(receipt)
        body.pop("artifact_sha256", None)
        if (
            supplied_sha == canonical_sha256(body)
            and all(receipt.get(name) == value for name, value in dependency.items())
        ):
            tensors = load_file(artifact_path, device="cpu")
            if all(
                receipt["tensor_sha256s"].get(name) == tensor_sha256(value)
                for name, value in tensors.items()
            ):
                emit("carrier-artifact-reused", artifact_sha256=supplied_sha)
                return tensors, receipt
    mapping = codebook.target_to_donor.clone().to(dtype=torch.int64)
    direct_lookup = torch.zeros(DONOR_COUNT, dtype=torch.bool)
    direct_lookup[mapping] = True
    non_direct_evidence_donors = sorted(
        donor_id
        for donor_id in evidence_counts
        if 0 <= donor_id < DONOR_COUNT and not bool(direct_lookup[donor_id])
    )
    if SELECTION_MODE in {
        "repository-corpus-v1",
        "natural-language-corpus-v1",
    }:
        relative_errors = reconstruction_relative_errors(
            donor_ids=non_direct_evidence_donors,
            offsets=codebook.semantic_code_offsets,
            target_ids=codebook.semantic_code_target_ids,
            coefficients=codebook.semantic_code_coefficients,
            parent_table=parent_table,
            target_table=original_target_table,
        )
        promotion_scores = {
            donor_id: evidence_counts[donor_id] * relative_errors[donor_id]
            for donor_id in non_direct_evidence_donors
        }
    else:
        promotion_scores = {
            donor_id: float(evidence_counts[donor_id])
            for donor_id in non_direct_evidence_donors
        }
    protected = {
        int(value)
        for value in getattr(donor_tokenizer, "all_special_ids", ())
        if isinstance(value, int) and 0 <= value < DONOR_COUNT
    }
    protected.update(cached_direct_winners(mapping))
    if SELECTION_MODE == "prompt-diagnostic-v1":
        envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
        protected.update(evidence_counts)
        protected.update(
            int(row["response_token_ids"][0])
            for row in envelope["responses"]
            if row["response_token_ids"]
        )
    priority, carriers, selection_report = select_carrier_reassignments(
        input_offsets=codebook.semantic_code_offsets,
        input_target_ids=codebook.semantic_code_target_ids,
        output_offsets=codebook.semantic_code_offsets,
        output_target_ids=codebook.semantic_code_target_ids,
        mapping=mapping,
        evidence_counts=evidence_counts,
        promotion_scores=promotion_scores,
        protected_donor_ids=protected,
        carrier_budget=CARRIER_BUDGET,
    )
    evicted = [int(mapping[target_id].item()) for target_id in carriers]
    final_table = original_target_table.clone()
    final_table[torch.tensor(carriers, dtype=torch.int64)] = parent_table.index_select(
        0, torch.tensor(priority, dtype=torch.int64)
    ).to(dtype=final_table.dtype)
    mapping[torch.tensor(carriers, dtype=torch.int64)] = torch.tensor(
        priority, dtype=torch.int64
    )
    supports, evicted_coefficients, omp_report = localized_residual_omp(
        source_rows=parent_table.index_select(
            0, torch.tensor(evicted, dtype=torch.int64)
        ),
        dictionary_rows=final_table,
        device=device,
    )
    evicted_supports = {
        donor_id: (supports[index].clone(), evicted_coefficients[index].clone())
        for index, donor_id in enumerate(evicted)
    }
    promoted_to_carrier = dict(zip(priority, carriers, strict=True))
    input_offsets, input_ids, input_coefficients = rewrite_code(
        base_offsets=codebook.semantic_code_offsets,
        base_target_ids=codebook.semantic_code_target_ids,
        base_coefficients=codebook.semantic_code_coefficients,
        promoted_to_carrier=promoted_to_carrier,
        evicted_supports=evicted_supports,
    )
    output_offsets, output_ids, final_output_coefficients = rewrite_code(
        base_offsets=codebook.semantic_code_offsets,
        base_target_ids=codebook.semantic_code_target_ids,
        base_coefficients=output_coefficients,
        promoted_to_carrier=promoted_to_carrier,
        evicted_supports=evicted_supports,
    )
    if not torch.equal(input_offsets, output_offsets) or not torch.equal(
        input_ids, output_ids
    ):
        raise RuntimeError("Carrier input/output CSR structures unexpectedly differ.")
    tensors = {
        "target_table": final_table.contiguous(),
        "target_to_donor": mapping.contiguous(),
        "input_semantic_code_offsets": input_offsets,
        "input_semantic_code_target_ids": input_ids,
        "input_semantic_code_coefficients": input_coefficients,
        "output_semantic_code_offsets": output_offsets,
        "output_semantic_code_target_ids": output_ids,
        "output_semantic_code_coefficients": final_output_coefficients,
    }
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    body = {
        "schema": "neural-forge-lfm2-semantic-carrier-replant-artifact-v2",
        "status": "complete",
        **dependency,
        **selection_report,
        **omp_report,
        "selection_evidence_review_prompt_tokens_used": bool(
            selection_evidence.get("review_prompt_tokens_used", False)
        ),
        "selection_evidence_parent_response_tokens_used": bool(
            selection_evidence.get("parent_response_tokens_used", False)
        ),
        "target_table_rows_reassigned": len(priority),
        "other_non_direct_rows_changed": 0,
        "reassigned_carriers_had_zero_prior_union_support_incidence": True,
        "spelling_channel_changed": False,
        "input_code_continuous_coefficient_count": int(input_coefficients.numel()),
        "output_code_continuous_coefficient_count": int(
            final_output_coefficients.numel()
        ),
        "tensor_sha256s": tensor_hashes,
        "operation_audit": {
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "global_map_solve_calls": 0,
            "full_dictionary_search_rows": len(evicted),
        },
    }
    receipt = {**body, "artifact_sha256": canonical_sha256(body)}
    save_file(tensors, artifact_path)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit(
        "carrier-artifact-compiled",
        artifact_sha256=receipt["artifact_sha256"],
        **selection_report,
        **omp_report,
    )
    return tensors, receipt


def progress_callback(value: dict[str, Any]) -> None:
    progress = value.get("runtime_progress", value)
    stage = progress.get("stage")
    if stage == "semantic-step-complete":
        step = int(progress["step_index"]) + 1
        if step % 16 and not progress.get("eos_selected"):
            return
    if stage in {
        "runtime-start",
        "prompt-start",
        "prompt-prefill-complete",
        "semantic-step-complete",
        "prompt-complete",
        "runtime-complete",
    }:
        details = dict(progress)
        details.pop("stage", None)
        emit("runtime-progress", runtime_stage=stage, **details)


def main() -> None:
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(RUNTIME_DEVICE_NAME)
    runtime_dtype = resolve_runtime_dtype(RUNTIME_DTYPE_NAME)
    routing_accumulation_dtype = (
        torch.float32
        if runtime_dtype in {torch.float16, torch.bfloat16}
        else runtime_dtype
    )
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    emit(
        "start",
        device=str(device),
        cuda_device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        runtime_dtype=str(runtime_dtype).removeprefix("torch."),
        routing_accumulation_dtype=str(routing_accumulation_dtype).removeprefix(
            "torch."
        ),
        localized_omp_selection_dtype="float32",
        localized_solve_dtype="float64-cpu",
        selection_mode=SELECTION_MODE,
        carrier_budget=CARRIER_BUDGET,
        carrier_allocation_algorithm_abi=CARRIER_ALLOCATION_ALGORITHM_ABI,
        localized_omp_algorithm_abi=LOCALIZED_OMP_ALGORITHM_ABI,
    )
    if file_sha256(OUTPUT_COEFFICIENTS) != OUTPUT_COEFFICIENTS_FILE_SHA:
        raise RuntimeError("Functional output coefficient checkpoint hash changed.")
    donor_tokenizer = load_lfm2_direct_local_tokenizer(DONOR_TOKENIZER)
    target_tokenizer = load_lfm2_direct_local_tokenizer(TARGET_TOKENIZER)
    bridge = load_strict_tokenizer_bridge_artifact(
        TARGET_TOKENIZER,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=BRIDGE_SHA,
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device("cpu"),
        dtype=runtime_dtype,
    )
    child, _ = checkpoint_io.load_authenticated_portable(
        CHECKPOINT,
        runtime_device=device,
        runtime_dtype=runtime_dtype,
    )
    child.requires_grad_(False)
    child.eval()
    codebook = load_lfm2_sparse_semantic_codebook(
        CODEBOOK, expected_artifact_sha256=CODEBOOK_SHA
    )
    output_bundle = load_file(OUTPUT_COEFFICIENTS, device="cpu")
    output_coefficients = output_bundle["output_semantic_code_coefficients"]
    with safe_open(PARENT / "model.safetensors", framework="pt", device="cpu") as source:
        parent_table = source.get_tensor("model.embed_tokens.weight")[:DONOR_COUNT]
    original_target_table = child.get_input_embeddings().weight.detach().to(
        device="cpu"
    )
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    tensors, artifact = compile_or_load_carrier_artifact(
        donor_tokenizer=donor_tokenizer,
        prompts=prompts,
        parent_table=parent_table,
        original_target_table=original_target_table,
        codebook=codebook,
        output_coefficients=output_coefficients,
        device=device,
    )
    live_table = child.get_input_embeddings().weight
    with torch.no_grad():
        live_table.copy_(tensors["target_table"].to(device=device, dtype=live_table.dtype))
    span = json.loads(SPAN.read_text(encoding="utf-8"))
    routing = bind_lfm2_separate_sparse_semantic_code_runtime(
        span,
        live_table,
        input_semantic_code_offsets=tensors["input_semantic_code_offsets"],
        input_semantic_code_target_ids=tensors["input_semantic_code_target_ids"],
        input_semantic_code_coefficients=tensors[
            "input_semantic_code_coefficients"
        ],
        output_semantic_code_offsets=tensors["output_semantic_code_offsets"],
        output_semantic_code_target_ids=tensors["output_semantic_code_target_ids"],
        output_semantic_code_coefficients=tensors[
            "output_semantic_code_coefficients"
        ],
        target_to_donor=tensors["target_to_donor"],
    )
    del parent_table, output_bundle, output_coefficients
    torch.cuda.empty_cache()
    response_directory = RUN_DIRECTORY / "prompt-results"
    response_directory.mkdir(exist_ok=True)
    prompt_results = []
    for index, prompt in enumerate(prompts):
        path = response_directory / f"{index:02d}-{prompt['id']}.json"
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("carrier_artifact_sha256") == artifact["artifact_sha256"]:
                prompt_results.append(cached)
                emit("prompt-result-reused", prompt_id=prompt["id"], index=index)
                continue
        result = run_lfm2_semantic_step_model_review(
            child,
            routing,
            donor_tokenizer,
            target_tokenizer,
            [prompt],
            maximum_prompt_tokens=8192,
            maximum_new_tokens=MAXIMUM_NEW_TOKENS,
            maximum_embedding_rows_per_chunk=1024,
            maximum_donor_rows_per_score_chunk=4096,
            progress_callback=progress_callback,
        )
        body = {
            "schema": "neural-forge-lfm2-carrier-replant-prompt-result-v1",
            "carrier_artifact_sha256": artifact["artifact_sha256"],
            "prompt_index": index,
            "prompt_id": prompt["id"],
            "runtime": result,
        }
        cached = {**body, "result_sha256": canonical_sha256(body)}
        path.write_text(
            json.dumps(cached, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        prompt_results.append(cached)
        emit(
            "prompt-result-complete",
            prompt_id=prompt["id"],
            index=index,
            result_sha256=cached["result_sha256"],
            response=result["responses"][0]["response"],
        )
    envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    parent_by_id = {row["id"]: row for row in envelope["responses"]}
    comparisons = []
    for prompt, prompt_result in zip(prompts, prompt_results, strict=True):
        child_response = prompt_result["runtime"]["responses"][0]
        comparisons.append(
            {
                "id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt,
                "parent_cached_response": parent_by_id[prompt["id"]],
                "child_response": child_response,
                "manual_review_instruction": (
                    "Judge coherence, correctness, instruction following, and useful "
                    "intelligence across the whole decoded response. Wording differences "
                    "and exact-token differences are not failures by themselves."
                ),
            }
        )
    physical_parameters = sum(
        int(parameter.numel()) for parameter in {id(p): p for p in child.parameters()}.values()
    )
    continuous = int(routing.continuous_semantic_code_coefficient_count)
    evaluation_counts = prompt_token_counts(donor_tokenizer, prompts)
    final_direct_donors = {
        int(value) for value in tensors["target_to_donor"].tolist()
    }
    exact_evaluation_occurrences = sum(
        count
        for donor_id, count in evaluation_counts.items()
        if donor_id in final_direct_donors
    )
    exact_evaluation_unique = sum(
        donor_id in final_direct_donors for donor_id in evaluation_counts
    )
    dossier_body = {
        "schema": "neural-forge-lfm2-carrier-replant-whole-response-dossier-v1",
        "status": "complete-awaiting-manual-whole-response-intelligence-review",
        "carrier_artifact_sha256": artifact["artifact_sha256"],
        "comparisons": comparisons,
        "prompt_count": len(comparisons),
        "parent_unique_parameters": 229_693_184,
        "physical_child_unique_parameters": physical_parameters,
        "routing_continuous_state_count": continuous,
        "effective_child_parameter_count": physical_parameters + continuous,
        "effective_parameter_compaction_ratio": 229_693_184
        / (physical_parameters + continuous),
        "training_steps": 0,
        "parent_forward_calls_during_compile_or_review": 0,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "automatic_acceptance_or_rejection_performed": False,
        "selection_mode": SELECTION_MODE,
        "selection_evidence_sha256": artifact["selection_evidence_sha256"],
        "selection_uses_review_prompt_tokens": artifact[
            "selection_evidence_review_prompt_tokens_used"
        ],
        "selection_uses_parent_response_tokens": artifact[
            "selection_evidence_parent_response_tokens_used"
        ],
        "evaluation_prompt_occurrence_count": sum(evaluation_counts.values()),
        "evaluation_prompt_unique_donor_count": len(evaluation_counts),
        "evaluation_prompt_exact_carrier_occurrence_count": (
            exact_evaluation_occurrences
        ),
        "evaluation_prompt_exact_carrier_unique_count": exact_evaluation_unique,
        "evaluation_coverage_measured_only_after_selection_was_frozen": True,
        "generalization_claimed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    dossier = {
        **dossier_body,
        "dossier_sha256": canonical_sha256(dossier_body),
    }
    dossier_path = RUN_DIRECTORY / "whole-response-dossier.json"
    dossier_path.write_text(
        json.dumps(dossier, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit(
        "complete",
        dossier=str(dossier_path),
        dossier_sha256=dossier["dossier_sha256"],
        elapsed_seconds=dossier["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
