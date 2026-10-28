from __future__ import annotations

import argparse
import copy
import json
import sys

sys.path.insert(0, "/home/cereal/model-surgery-lab/neural-forge-src")

import torch

from experiments.run_lfm2_cpcf_blank_fold import _trace_adapter
from experiments.run_lfm2_residual_attention_fold import (
    load_local_lfm2_model,
    load_local_tokenizer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_installability import (
    Lfm2CpcfInstallabilityAdapter,
)
from neural_forge.hf_lfm2_cpcf_layer_compiler import (
    _phase_evidence,
    _rope_terms,
    _variance_epsilon,
    _weight,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    build_lfm2_cpcf_target_phases,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    _apply_rope,
    _cached_lfm2_rms_norm,
    _relative_rmse,
    _rope_pair_relative_error,
)
from neural_forge.hf_tokenizer_bridge import (
    load_strict_tokenizer_bridge_artifact,
)


PARENT = (
    "/home/cereal/.cache/huggingface/hub/"
    "models--LiquidAI--LFM2.5-230M/snapshots/"
    "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7"
)
TARGET_TOKENIZER = (
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2"
)
CACHE = (
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_tokenizer_bridge_evidence_20260717/"
    "v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/"
    "76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"
)
CORPUS = (
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_bridge_folding_corpora_20260717/"
    "v4096_broad256_v1/trajectories.json"
)


def metric(
    normalized: torch.Tensor,
    rotated: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> dict[str, float]:
    reconstruction = _apply_rope(normalized, cosine, sine)
    return {
        "pair_relative_error": _rope_pair_relative_error(
            normalized,
            rotated,
            reconstruction,
        ),
        "relative_rmse": _relative_rmse(reconstruction, rotated),
        "maximum_absolute_error": float(
            (reconstruction - rotated).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-ordinal", type=int, default=54)
    parser.add_argument("--list-splits", action="store_true")
    args = parser.parse_args()
    with open(CORPUS, encoding="utf-8") as handle:
        rows = json.load(handle)
    if args.list_splits:
        print(
            json.dumps(
                {
                    split: [
                        index
                        for index, row in enumerate(rows)
                        if row.get("split") == split
                    ]
                    for split in ("selection", "fit", "validation")
                },
                sort_keys=True,
            )
        )
        return
    fit_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("split") == "fit"
    ]
    batch_index = fit_indices[args.fit_ordinal]
    base = open_lfm2_circuit_trace_cache(CACHE, verify=False)
    manifest = copy.deepcopy(base.manifest)
    manifest["corpus"]["batch_count"] = 1
    manifest["corpus"]["ordered_batch_input_sha256"] = [
        base.manifest["corpus"]["ordered_batch_input_sha256"][batch_index]
    ]

    class SubsetCache:
        cache_key = base.cache_key
        batch_count = 1

        def __init__(self) -> None:
            self.manifest = manifest

        def verify(self) -> SubsetCache:
            base.load_batch(batch_index)
            return self

        def load_batch(self, _ordinal: int) -> object:
            return base.load_batch(batch_index)

    donor_tokenizer = load_local_tokenizer(PARENT)
    target_tokenizer = load_local_tokenizer(TARGET_TOKENIZER)
    bridge = load_strict_tokenizer_bridge_artifact(
        TARGET_TOKENIZER,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    trace = _trace_adapter(
        rows=[rows[batch_index]],
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache_manifest=manifest,
    )
    phases = build_lfm2_cpcf_target_phases(
        trace,
        Lfm2CpcfInstallabilityAdapter(SubsetCache()).iter_compiler_inputs(),
    )
    parent = load_local_lfm2_model(
        PARENT,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
    )
    parent.requires_grad_(False)
    parent.eval()
    epsilon = float(torch.finfo(torch.bfloat16).eps)
    results = []
    for layer_index, layer in enumerate(parent.model.layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        q_weight = _weight(
            attention.q_layernorm,
            "source_attention.q_layernorm",
            dimensions=1,
        )
        k_weight = _weight(
            attention.k_layernorm,
            "source_attention.k_layernorm",
            dimensions=1,
        )
        q_norm_epsilon = _variance_epsilon(
            attention.q_layernorm,
            "source_attention.q_layernorm",
        )
        k_norm_epsilon = _variance_epsilon(
            attention.k_layernorm,
            "source_attention.k_layernorm",
        )
        q_linear_parts = []
        q_parts = []
        q_cached_norm_parts = []
        k_linear_parts = []
        k_parts = []
        k_cached_norm_parts = []
        cached_cos_parts = []
        cached_sin_parts = []
        for phase in phases:
            tensors = _phase_evidence(
                phase,
                layer_index,
                "full_attention",
            )
            q_linear_parts.append(tensors["attention.q_linear"])
            q_parts.append(tensors["attention.q"])
            q_cached_norm_parts.append(
                tensors["attention.q_normalized_pre_rope"]
            )
            k_linear_parts.append(tensors["attention.k_linear"])
            k_parts.append(tensors["attention.k"])
            k_cached_norm_parts.append(
                tensors["attention.k_normalized_pre_rope"]
            )
            cached_cos_parts.append(tensors["attention.rope_cos"])
            cached_sin_parts.append(tensors["attention.rope_sin"])
        q_linear = torch.cat(q_linear_parts, dim=2)
        q_rotated = torch.cat(q_parts, dim=2).to(torch.float64)
        q_cached_norm = torch.cat(q_cached_norm_parts, dim=2).to(
            torch.float64
        )
        k_linear = torch.cat(k_linear_parts, dim=2)
        k_rotated = torch.cat(k_parts, dim=2).to(torch.float64)
        k_cached_norm = torch.cat(k_cached_norm_parts, dim=2).to(
            torch.float64
        )
        cached_cos = torch.cat(cached_cos_parts, dim=1).to(torch.float64)
        cached_sin = torch.cat(cached_sin_parts, dim=1).to(torch.float64)
        static_cos, static_sin = _rope_terms(
            parent,
            phases[-1].position_ids,
            dtype=q_linear.dtype,
        )
        static_cos = static_cos.to(torch.float64)
        static_sin = static_sin.to(torch.float64)
        q_reconstructed_norm = _cached_lfm2_rms_norm(
            q_linear,
            q_weight,
            q_norm_epsilon,
        )
        k_reconstructed_norm = _cached_lfm2_rms_norm(
            k_linear,
            k_weight,
            k_norm_epsilon,
        )
        result = {
            "layer_index": layer_index,
            "epsilon": epsilon,
            "q_reconstructed_cached_norm_pair_error": (
                _rope_pair_relative_error(
                    q_cached_norm,
                    q_cached_norm,
                    q_reconstructed_norm,
                )
            ),
            "k_reconstructed_cached_norm_pair_error": (
                _rope_pair_relative_error(
                    k_cached_norm,
                    k_cached_norm,
                    k_reconstructed_norm,
                )
            ),
            "q_static": metric(
                q_reconstructed_norm,
                q_rotated,
                static_cos,
                static_sin,
            ),
            "q_cached_rope": metric(
                q_reconstructed_norm,
                q_rotated,
                cached_cos,
                cached_sin,
            ),
            "q_cached_norm_and_rope": metric(
                q_cached_norm,
                q_rotated,
                cached_cos,
                cached_sin,
            ),
            "k_static": metric(
                k_reconstructed_norm,
                k_rotated,
                static_cos,
                static_sin,
            ),
            "k_cached_rope": metric(
                k_reconstructed_norm,
                k_rotated,
                cached_cos,
                cached_sin,
            ),
            "k_cached_norm_and_rope": metric(
                k_cached_norm,
                k_rotated,
                cached_cos,
                cached_sin,
            ),
            "static_cached_cos_max_abs": float(
                (static_cos - cached_cos).abs().max().item()
            ),
            "static_cached_sin_max_abs": float(
                (static_sin - cached_sin).abs().max().item()
            ),
        }
        for name in (
            "q_static",
            "q_cached_rope",
            "q_cached_norm_and_rope",
            "k_static",
            "k_cached_rope",
            "k_cached_norm_and_rope",
        ):
            result[name]["error_in_eps"] = (
                result[name]["pair_relative_error"] / epsilon
            )
        results.append(result)
    print(
        json.dumps(
            {
                "fit_ordinal": args.fit_ordinal,
                "batch_index": batch_index,
                "record_id": rows[batch_index].get("record_id"),
                "phase_count": len(phases),
                "results": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
