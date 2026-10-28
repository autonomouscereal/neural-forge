from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


RUNTIME = Path("/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2")
sys.path.insert(0, str(RUNTIME / "src"))

from neural_forge.hf_lfm2_circuit_trace_cache import Lfm2CircuitTraceCache  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_model_runtime import (  # noqa: E402
    _body_output,
    _donor_prompt_ids,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    bind_lfm2_sparse_semantic_code_runtime,
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
SPAN = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/"
    "lfm2-semantic-step-global-raw-span/"
    "3e780d07ac1034ee77756f973b483637cee846e17e6d7191d6b20c9533360167/"
    "payload/span-evidence.json"
)
TRACE = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/"
    "parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b"
)
PROMPTS = RUNTIME / "research/lfm2_semantic_whole_response_prompts_v1.json"
ENVELOPE = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/"
    "e739a88/a0002/parent-envelope.json"
)
RUN_DIRECTORY = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "neural_forge_functional_output_refit_20260730/a5-fixed-a4-support"
)
DONOR_COUNT = 64_402
TARGET_COUNT = 4_096
HIDDEN_SIZE = 1_024
FIT_HOLDOUT_MODULUS = 4
FIT_HOLDOUT_REMAINDER = 3
FUNCTIONAL_WEIGHTS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
SOLVE_RIDGE = 1e-4
SOLVE_ROWS_PER_CHUNK = 512


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


def head_logit_pairs(names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    inventory = frozenset(names)
    suffix = ".output.lm_head_input"
    result = []
    for hidden_name in sorted(name for name in inventory if name.endswith(suffix)):
        phase = hidden_name[: -len(suffix)]
        full = f"{phase}.full_prefix_logits"
        legacy = f"{phase}.logits"
        if legacy in inventory:
            result.append((hidden_name, legacy))
        elif full in inventory:
            result.append((hidden_name, full))
    return tuple(result)


def load_cached_hidden_rows() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    cache = Lfm2CircuitTraceCache.open(TRACE, verify=False)
    hidden_rows = []
    parent_winners = []
    batch_ids = []
    pair_count = 0
    logical_bytes = 0
    with cache.selective_authenticated_session() as session:
        for reference in session.batch_references:
            pairs = head_logit_pairs(reference.ordered_tensor_names)
            if not pairs:
                continue
            selected_names = tuple(sorted({name for pair in pairs for name in pair}))
            payload = session.load_selected_tensors(reference.index, selected_names)
            try:
                logical_bytes += int(payload.logical_bytes)
                for hidden_name, logit_name in pairs:
                    hidden = payload.tensors[hidden_name]
                    logits = payload.tensors[logit_name]
                    flat_hidden = hidden.reshape(-1, HIDDEN_SIZE).to(
                        device="cpu", dtype=torch.float32
                    )
                    flat_logits = logits[..., :DONOR_COUNT].reshape(
                        -1, DONOR_COUNT
                    )
                    if int(flat_logits.shape[0]) != int(flat_hidden.shape[0]):
                        raise RuntimeError("Cached hidden/logit rows are unaligned.")
                    hidden_rows.append(flat_hidden)
                    parent_winners.append(
                        flat_logits.argmax(dim=-1).to(device="cpu", dtype=torch.int64)
                    )
                    batch_ids.append(
                        torch.full(
                            (int(flat_hidden.shape[0]),),
                            int(reference.index),
                            dtype=torch.int64,
                        )
                    )
                    pair_count += 1
            finally:
                session.release_selected_tensors(payload)
        session.verify()
        session_receipt = session.as_dict()
    hidden = torch.cat(hidden_rows, dim=0)
    winners = torch.cat(parent_winners, dim=0)
    batches = torch.cat(batch_ids, dim=0)
    return hidden, winners, batches, {
        "trace_cache_key": cache.cache_key,
        "trace_parent_fingerprint": cache.parent_fingerprint,
        "trace_selected_pair_count": pair_count,
        "trace_selected_logical_bytes": logical_bytes,
        "trace_selective_session_sha256": session_receipt["session_sha256"],
    }


def solve_output_coefficients(
    *,
    metric: torch.Tensor,
    functional_weight: float,
    donor_table: torch.Tensor,
    target_table: torch.Tensor,
    offsets: torch.Tensor,
    target_ids: torch.Tensor,
    base_coefficients: torch.Tensor,
    target_to_donor: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    if metric.dtype != torch.float32 or metric.device != device:
        raise RuntimeError("Functional metric placement is invalid.")
    identity = torch.eye(HIDDEN_SIZE, device=device, dtype=torch.float32)
    combined_metric = metric + float(functional_weight) * identity
    target_metric = target_table @ combined_metric
    lengths = offsets[1:] - offsets[:-1]
    direct_mask = torch.zeros(DONOR_COUNT, dtype=torch.bool)
    direct_mask[target_to_donor] = True
    non_direct = torch.nonzero(~direct_mask, as_tuple=False).reshape(-1)
    if not bool(torch.all(lengths[direct_mask] == 1)) or not bool(
        torch.all(lengths[non_direct] == 16)
    ):
        raise RuntimeError("A4 support cardinalities changed unexpectedly.")
    positions = offsets.index_select(0, non_direct).unsqueeze(1) + torch.arange(
        16, dtype=torch.int64
    ).unsqueeze(0)
    support_ids = target_ids.index_select(0, positions.reshape(-1)).reshape(-1, 16)
    coefficients = torch.zeros(int(target_ids.numel()), dtype=torch.float32)
    direct_positions = offsets.index_select(0, target_to_donor)
    coefficients.index_fill_(0, direct_positions, 1.0)
    solve_failure_count = 0
    maximum_solve_residual = 0.0
    coefficient_abs_maximum = 0.0
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, int(non_direct.numel()), SOLVE_ROWS_PER_CHUNK):
            stop = min(start + SOLVE_ROWS_PER_CHUNK, int(non_direct.numel()))
            donor_chunk_ids = non_direct[start:stop].to(device=device)
            support_chunk = support_ids[start:stop].to(device=device)
            basis = target_table[support_chunk]
            basis_metric = target_metric[support_chunk]
            gram = torch.bmm(basis_metric, basis.transpose(1, 2))
            gram.diagonal(dim1=-2, dim2=-1).add_(SOLVE_RIDGE)
            source = donor_table.index_select(0, donor_chunk_ids)
            rhs = torch.bmm(basis_metric, source.unsqueeze(2))
            base = base_coefficients.index_select(
                0, positions[start:stop].reshape(-1)
            ).reshape(stop - start, 16, 1).to(device=device)
            rhs.add_(base, alpha=SOLVE_RIDGE)
            solved, info = torch.linalg.solve_ex(gram, rhs, check_errors=False)
            failed = info.ne(0) | ~torch.isfinite(solved).all(dim=(1, 2))
            if bool(failed.any()):
                solve_failure_count += int(failed.sum().item())
                fallback = torch.linalg.lstsq(
                    gram[failed], rhs[failed], driver="gels"
                ).solution
                solved[failed] = fallback
            residual = torch.bmm(gram, solved) - rhs
            maximum_solve_residual = max(
                maximum_solve_residual,
                float(torch.linalg.vector_norm(residual, dim=(1, 2)).max().item()),
            )
            flat = solved.squeeze(2).to(device="cpu", dtype=torch.float32)
            coefficient_abs_maximum = max(
                coefficient_abs_maximum, float(flat.abs().max().item())
            )
            coefficient_positions = positions[start:stop].reshape(-1)
            coefficients.index_copy_(0, coefficient_positions, flat.reshape(-1))
            if start == 0 or stop == int(non_direct.numel()) or stop % 8192 < SOLVE_ROWS_PER_CHUNK:
                emit(
                    "solve-progress",
                    functional_weight=functional_weight,
                    completed_non_direct_rows=stop,
                    total_non_direct_rows=int(non_direct.numel()),
                    elapsed_seconds=time.monotonic() - started,
                    solve_failure_count=solve_failure_count,
                )
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("Functional output coefficients are nonfinite.")
    return coefficients, {
        "functional_weight": functional_weight,
        "solve_ridge": SOLVE_RIDGE,
        "solve_rows_per_chunk": SOLVE_ROWS_PER_CHUNK,
        "solve_failure_count": solve_failure_count,
        "maximum_solve_residual": maximum_solve_residual,
        "coefficient_abs_maximum": coefficient_abs_maximum,
        "elapsed_seconds": time.monotonic() - started,
    }


def sparse_design(
    offsets: torch.Tensor,
    target_ids: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    lengths = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(torch.arange(DONOR_COUNT), lengths)
    return torch.sparse_coo_tensor(
        torch.stack((rows, target_ids)),
        coefficients,
        size=(DONOR_COUNT, TARGET_COUNT),
        dtype=torch.float32,
        device="cpu",
        check_invariants=True,
    ).coalesce().to(device)


def top1_report(
    *,
    design: torch.Tensor,
    hidden: torch.Tensor,
    winners: torch.Tensor,
    batches: torch.Tensor,
    target_table: torch.Tensor,
    target_to_donor: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    direct_lookup = torch.zeros(DONOR_COUNT, dtype=torch.bool)
    direct_lookup[target_to_donor] = True
    predictions = []
    for start in range(0, int(hidden.shape[0]), 64):
        chunk = hidden[start : start + 64].to(device=device, dtype=torch.float32)
        target_scores = chunk @ target_table.transpose(0, 1)
        logits = torch.sparse.mm(design, target_scores.transpose(0, 1)).transpose(0, 1)
        predictions.append(logits.argmax(dim=-1).to(device="cpu"))
    predicted = torch.cat(predictions)
    matched = predicted.eq(winners)
    holdout = batches.remainder(FIT_HOLDOUT_MODULUS).eq(FIT_HOLDOUT_REMAINDER)
    parent_direct = direct_lookup.index_select(0, winners)

    def count(mask: torch.Tensor) -> dict[str, object]:
        total = int(mask.sum().item())
        matches = int((matched & mask).sum().item())
        return {
            "row_count": total,
            "top1_match_count": matches,
            "top1_match_rate": None if total == 0 else matches / total,
        }

    return {
        "all": count(torch.ones_like(matched, dtype=torch.bool)),
        "fit": count(~holdout),
        "holdout": count(holdout),
        "parent_direct": count(parent_direct),
        "parent_multi": count(~parent_direct),
    }


def prompt_hidden_rows(
    *,
    child: object,
    input_routing: object,
    live_table: torch.Tensor,
    exact_table_runtime: torch.Tensor,
    donor_tokenizer: object,
    prompts: list[dict[str, object]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    exact_hidden_rows = []
    folded_hidden_rows = []
    exact_winners = []
    prompt_ids = []
    body = child.model
    with torch.inference_mode(), input_routing.authenticated_live_table_session(
        live_table
    ) as session:
        for prompt in prompts:
            ids = _donor_prompt_ids(donor_tokenizer, prompt)[0].tolist()
            index = torch.tensor(ids, device=device, dtype=torch.int64)
            exact_embeddings = exact_table_runtime.index_select(0, index)
            folded_embeddings = input_routing.reconstruct_selected_donor_embeddings(
                session, ids, maximum_rows_per_chunk=1024
            )
            attention = torch.ones((1, len(ids)), device=device, dtype=torch.int64)
            exact_hidden, _ = _body_output(
                body(
                    inputs_embeds=exact_embeddings.unsqueeze(0),
                    attention_mask=attention,
                    use_cache=True,
                ),
                expected_hidden_size=HIDDEN_SIZE,
            )
            folded_hidden, _ = _body_output(
                body(
                    inputs_embeds=folded_embeddings.unsqueeze(0),
                    attention_mask=attention,
                    use_cache=True,
                ),
                expected_hidden_size=HIDDEN_SIZE,
            )
            exact_last = exact_hidden[:, -1, :].float()
            folded_last = folded_hidden[:, -1, :].float()
            exact_hidden_rows.append(exact_last.to(device="cpu"))
            folded_hidden_rows.append(folded_last.to(device="cpu"))
            exact_winners.append(
                int((exact_last @ exact_table_runtime.float().transpose(0, 1)).argmax())
            )
            prompt_ids.append(str(prompt["id"]))
            emit("prompt-hidden-complete", prompt_id=prompt["id"])
    return (
        torch.cat(exact_hidden_rows),
        torch.cat(folded_hidden_rows),
        torch.tensor(exact_winners, dtype=torch.int64),
        prompt_ids,
    )


def prompt_report(
    *,
    design: torch.Tensor,
    exact_hidden: torch.Tensor,
    folded_hidden: torch.Tensor,
    exact_winners: torch.Tensor,
    envelope_winners: torch.Tensor,
    target_table: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    def predict(hidden: torch.Tensor) -> torch.Tensor:
        target_scores = hidden.to(device=device) @ target_table.transpose(0, 1)
        return torch.sparse.mm(design, target_scores.transpose(0, 1)).transpose(
            0, 1
        ).argmax(dim=-1).to(device="cpu")

    exact_input = predict(exact_hidden)
    folded_input = predict(folded_hidden)
    return {
        "exact_input_output_code_match_to_exact_winner_count": int(
            exact_input.eq(exact_winners).sum().item()
        ),
        "folded_input_output_code_match_to_exact_winner_count": int(
            folded_input.eq(exact_winners).sum().item()
        ),
        "exact_input_output_code_match_to_envelope_count": int(
            exact_input.eq(envelope_winners).sum().item()
        ),
        "folded_input_output_code_match_to_envelope_count": int(
            folded_input.eq(envelope_winners).sum().item()
        ),
        "exact_input_predictions": exact_input.tolist(),
        "folded_input_predictions": folded_input.tolist(),
    }


def main() -> None:
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    started = time.monotonic()
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    emit(
        "start",
        device=str(device),
        torch_version=torch.__version__,
        cuda_device_name=torch.cuda.get_device_name(device),
        solve_dtype="float32",
        published_coefficient_dtype="float32",
    )
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
        dtype=torch.bfloat16,
    )
    child, _ = checkpoint_io.load_authenticated_portable(
        CHECKPOINT,
        runtime_device=device,
        runtime_dtype=torch.bfloat16,
    )
    child.requires_grad_(False)
    child.eval()
    with safe_open(PARENT / "model.safetensors", framework="pt", device="cpu") as source:
        donor_table_cpu = source.get_tensor("model.embed_tokens.weight")[:DONOR_COUNT]
    donor_table = donor_table_cpu.to(device=device, dtype=torch.float32)
    exact_table_runtime = donor_table_cpu.to(device=device, dtype=torch.bfloat16)
    target_table_runtime = child.get_input_embeddings().weight
    target_table = target_table_runtime.detach().to(dtype=torch.float32)
    codebook = load_lfm2_sparse_semantic_codebook(
        CODEBOOK, expected_artifact_sha256=CODEBOOK_SHA
    )
    offsets = codebook.semantic_code_offsets.to(dtype=torch.int64)
    target_ids = codebook.semantic_code_target_ids.to(dtype=torch.int64)
    target_to_donor = codebook.target_to_donor.to(dtype=torch.int64)
    span = json.loads(SPAN.read_text(encoding="utf-8"))
    input_routing = bind_lfm2_sparse_semantic_code_runtime(
        span,
        target_table_runtime,
        semantic_code_offsets=offsets,
        semantic_code_target_ids=target_ids,
        semantic_code_coefficients=codebook.semantic_code_coefficients,
        target_to_donor=target_to_donor,
    )
    hidden, winners, batches, trace_receipt = load_cached_hidden_rows()
    holdout = batches.remainder(FIT_HOLDOUT_MODULUS).eq(FIT_HOLDOUT_REMAINDER)
    fit_hidden = hidden[~holdout].to(device=device, dtype=torch.float32)
    normalized = torch.nn.functional.normalize(fit_hidden, dim=1) * math.sqrt(
        HIDDEN_SIZE
    )
    metric = normalized.transpose(0, 1) @ normalized / int(normalized.shape[0])
    metric = (metric + metric.transpose(0, 1)) * 0.5
    metric_trace = float(torch.trace(metric).item())
    metric_sha256 = tensor_sha256(metric)
    implementation_sha256 = file_sha256(Path(__file__).resolve())
    emit(
        "evidence-ready",
        cached_hidden_rows=int(hidden.shape[0]),
        fit_rows=int((~holdout).sum().item()),
        holdout_rows=int(holdout.sum().item()),
        functional_metric_trace=metric_trace,
        functional_metric_sha256=metric_sha256,
        implementation_sha256=implementation_sha256,
        **trace_receipt,
    )
    prompts_document = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompts = prompts_document["prompts"]
    envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    envelope_by_id = {row["id"]: row for row in envelope["responses"]}
    prompt_exact_hidden, prompt_folded_hidden, prompt_exact_winners, prompt_ids = (
        prompt_hidden_rows(
            child=child,
            input_routing=input_routing,
            live_table=target_table_runtime,
            exact_table_runtime=exact_table_runtime,
            donor_tokenizer=donor_tokenizer,
            prompts=prompts,
            device=device,
        )
    )
    envelope_winners = torch.tensor(
        [int(envelope_by_id[prompt_id]["response_token_ids"][0]) for prompt_id in prompt_ids],
        dtype=torch.int64,
    )
    candidates = []
    for functional_weight in FUNCTIONAL_WEIGHTS:
        label = str(functional_weight).replace(".", "p")
        tensor_path = RUN_DIRECTORY / f"output-coefficients-lambda-{label}.safetensors"
        checkpoint_path = RUN_DIRECTORY / f"candidate-lambda-{label}.json"
        if tensor_path.is_file() and checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                checkpoint.get("schema")
                == "neural-forge-lfm2-functional-output-refit-candidate-v1"
                and checkpoint.get("functional_weight") == functional_weight
                and checkpoint.get("input_codebook_sha256") == CODEBOOK_SHA
                and checkpoint.get("functional_metric_sha256") == metric_sha256
                and checkpoint.get("implementation_sha256") == implementation_sha256
                and checkpoint.get("coefficient_file_sha256")
                == file_sha256(tensor_path)
            ):
                candidates.append(checkpoint)
                emit("candidate-reused", **checkpoint)
                continue
        coefficients, solve_report = solve_output_coefficients(
            metric=metric,
            functional_weight=functional_weight,
            donor_table=donor_table,
            target_table=target_table,
            offsets=offsets,
            target_ids=target_ids,
            base_coefficients=codebook.semantic_code_coefficients,
            target_to_donor=target_to_donor,
            device=device,
        )
        design = sparse_design(
            offsets, target_ids, coefficients, device=device
        )
        cached_report = top1_report(
            design=design,
            hidden=hidden,
            winners=winners,
            batches=batches,
            target_table=target_table,
            target_to_donor=target_to_donor,
            device=device,
        )
        first_token_report = prompt_report(
            design=design,
            exact_hidden=prompt_exact_hidden,
            folded_hidden=prompt_folded_hidden,
            exact_winners=prompt_exact_winners,
            envelope_winners=envelope_winners,
            target_table=target_table,
            device=device,
        )
        save_file(
            {"output_semantic_code_coefficients": coefficients.contiguous()},
            tensor_path,
        )
        report = {
            "schema": "neural-forge-lfm2-functional-output-refit-candidate-v1",
            "input_codebook_sha256": CODEBOOK_SHA,
            "functional_metric_sha256": metric_sha256,
            "implementation_sha256": implementation_sha256,
            **solve_report,
            "coefficient_count": int(coefficients.numel()),
            "coefficient_sha256": tensor_sha256(coefficients),
            "coefficient_file": tensor_path.name,
            "coefficient_file_sha256": file_sha256(tensor_path),
            "cached_hidden": cached_report,
            "first_token": first_token_report,
        }
        checkpoint_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        candidates.append(report)
        emit("candidate-complete", **report)
        del design
        torch.cuda.empty_cache()
    result = {
        "schema": "neural-forge-lfm2-functional-output-refit-prototype-v1",
        "status": "complete-advisory-no-automatic-selection",
        "elapsed_seconds": time.monotonic() - started,
        "input_codebook_sha256": CODEBOOK_SHA,
        "support_ids_reused_without_candidate_search": True,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "functional_metric_fit_rows": int((~holdout).sum().item()),
        "functional_metric_holdout_rows": int(holdout.sum().item()),
        "functional_metric_trace": metric_trace,
        "functional_metric_sha256": metric_sha256,
        "implementation_sha256": implementation_sha256,
        "candidates": candidates,
    }
    result_path = RUN_DIRECTORY / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    emit("complete", result_file=str(result_path), result_file_sha256=file_sha256(result_path), **result)


if __name__ == "__main__":
    main()
