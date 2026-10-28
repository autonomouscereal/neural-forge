from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


RUNTIME = Path("/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2")
sys.path.insert(0, str(RUNTIME / "src"))

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
BRIDGE = TARGET_TOKENIZER
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
PROMPTS = RUNTIME / "research/lfm2_semantic_whole_response_prompts_v1.json"
ENVELOPE = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/"
    "e739a88/a0002/parent-envelope.json"
)


def centered_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean(dim=-1, keepdim=True)
    b = b.float() - b.float().mean(dim=-1, keepdim=True)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).item())


def main() -> None:
    device = torch.device("cuda:0")
    runtime_dtype = torch.bfloat16
    donor_tokenizer = load_lfm2_direct_local_tokenizer(DONOR_TOKENIZER)
    target_tokenizer = load_lfm2_direct_local_tokenizer(TARGET_TOKENIZER)
    bridge = load_strict_tokenizer_bridge_artifact(
        BRIDGE,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=BRIDGE_SHA,
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    child, _receipt = checkpoint_io.load_authenticated_portable(
        CHECKPOINT,
        runtime_device=device,
        runtime_dtype=runtime_dtype,
    )
    child.requires_grad_(False)
    child.eval()
    with safe_open(PARENT / "model.safetensors", framework="pt", device="cpu") as source:
        exact_table = source.get_tensor("model.embed_tokens.weight")[:64402]
    exact_table = exact_table.to(device=device, dtype=runtime_dtype)
    codebook = load_lfm2_sparse_semantic_codebook(
        CODEBOOK,
        expected_artifact_sha256=CODEBOOK_SHA,
    )
    span = json.loads(SPAN.read_text(encoding="utf-8"))
    live_table = child.get_input_embeddings().weight
    routing = bind_lfm2_sparse_semantic_code_runtime(
        span,
        live_table,
        semantic_code_offsets=codebook.semantic_code_offsets,
        semantic_code_target_ids=codebook.semantic_code_target_ids,
        semantic_code_coefficients=codebook.semantic_code_coefficients,
        target_to_donor=codebook.target_to_donor,
    )
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
    if isinstance(prompts, dict):
        prompts = prompts.get("prompts", prompts.get("rows"))
    envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    parent_rows = envelope.get("responses", envelope.get("comparisons"))
    parent_by_id = {row["id"]: row for row in parent_rows}
    body = child.model
    counts = {"ee": 0, "ef": 0, "fe": 0, "ff": 0}
    reports = []
    with torch.inference_mode(), routing.authenticated_live_table_session(live_table) as session:
        for prompt in prompts:
            ids = _donor_prompt_ids(donor_tokenizer, prompt)[0].tolist()
            exact_embeddings = exact_table.index_select(
                0, torch.tensor(ids, device=device, dtype=torch.int64)
            )
            folded_embeddings = routing.reconstruct_selected_donor_embeddings(
                session, ids, maximum_rows_per_chunk=1024
            )
            attention = torch.ones((1, len(ids)), device=device, dtype=torch.int64)
            exact_hidden, _ = _body_output(
                body(
                    inputs_embeds=exact_embeddings.unsqueeze(0),
                    attention_mask=attention,
                    use_cache=True,
                ),
                expected_hidden_size=exact_table.shape[1],
            )
            folded_hidden, _ = _body_output(
                body(
                    inputs_embeds=folded_embeddings.unsqueeze(0),
                    attention_mask=attention,
                    use_cache=True,
                ),
                expected_hidden_size=exact_table.shape[1],
            )
            he = exact_hidden[:, -1, :]
            hf = folded_hidden[:, -1, :]
            exact_on_exact = torch.nn.functional.linear(he, exact_table)
            exact_on_folded = torch.nn.functional.linear(hf, exact_table)
            target_on_exact = routing.score_target_tokens(session, he)
            target_on_folded = routing.score_target_tokens(session, hf)
            folded_on_exact = routing.compute_all_donor_logits_from_target_scores(
                session, target_on_exact, maximum_donor_rows_per_chunk=4096
            )
            folded_on_folded = routing.compute_all_donor_logits_from_target_scores(
                session, target_on_folded, maximum_donor_rows_per_chunk=4096
            )
            parent_tokens = parent_by_id[prompt["id"]].get(
                "response_token_ids",
                parent_by_id[prompt["id"]].get("raw_donor_response_token_ids"),
            )
            parent_first = int(parent_tokens[0])
            logits = {
                "ee": exact_on_exact,
                "ef": folded_on_exact,
                "fe": exact_on_folded,
                "ff": folded_on_folded,
            }
            tops = {name: int(value.argmax(dim=-1).item()) for name, value in logits.items()}
            for name, token in tops.items():
                counts[name] += int(token == parent_first)
            reports.append(
                {
                    "id": prompt["id"],
                    "parent_first": parent_first,
                    "top1": tops,
                    "matches": {name: token == parent_first for name, token in tops.items()},
                    "prompt_last_hidden_cosine": float(
                        torch.nn.functional.cosine_similarity(he.float(), hf.float(), dim=-1).item()
                    ),
                    "prompt_last_hidden_relative_l2": float(
                        (torch.linalg.vector_norm(he.float() - hf.float()) / torch.linalg.vector_norm(he.float())).item()
                    ),
                    "output_only_centered_logit_cosine": centered_cosine(exact_on_exact, folded_on_exact),
                    "input_only_centered_logit_cosine": centered_cosine(exact_on_exact, exact_on_folded),
                    "combined_centered_logit_cosine": centered_cosine(exact_on_exact, folded_on_folded),
                }
            )
            print(json.dumps({"stage": "prompt-complete", **reports[-1]}), flush=True)
    result = {
        "schema": "neural-forge-lfm2-a4-first-token-isolation-prototype-v1",
        "lane_meanings": {
            "ee": "exact-input-exact-readout",
            "ef": "exact-input-folded-readout",
            "fe": "folded-input-exact-readout",
            "ff": "folded-input-folded-readout",
        },
        "first_token_matches": counts,
        "prompt_count": len(reports),
        "reports": reports,
    }
    print(json.dumps({"stage": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
