from __future__ import annotations

import copy
import json
import sys

sys.path.insert(0, "/home/cereal/model-surgery-lab/neural-forge-src")

import torch

from experiments.run_lfm2_cpcf_blank_fold import (
    _mapped_source_tied_table,
    _trace_adapter,
)
from experiments.run_lfm2_residual_attention_fold import (
    load_local_lfm2_model,
    load_local_tokenizer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_blank_target import (
    plan_and_synthesize_lfm2_cpcf_blank,
)
from neural_forge.hf_lfm2_cpcf_installability import (
    Lfm2CpcfInstallabilityAdapter,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    build_lfm2_cpcf_target_phases,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    Lfm2CpcfModelFoldPolicy,
    compile_lfm2_cpcf_model_fold,
)
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    Lfm2CpcfShortConvGatePolicy,
)
from neural_forge.hf_tokenizer_bridge import (
    fingerprint_hf_tokenizer_semantics,
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
INDICES = (0, 92, 98)


def stage(name: str, **values: object) -> None:
    print(json.dumps({"stage": name, **values}, sort_keys=True), flush=True)


def main() -> None:
    base = open_lfm2_circuit_trace_cache(CACHE, verify=False)
    manifest = base.manifest
    selected_hashes = [
        manifest["corpus"]["ordered_batch_input_sha256"][index]
        for index in INDICES
    ]

    class SubsetCache:
        def __init__(self) -> None:
            self.cache_key = base.cache_key
            self.batch_count = len(INDICES)
            self.manifest = copy.deepcopy(manifest)
            self.manifest["corpus"]["batch_count"] = len(INDICES)
            self.manifest["corpus"][
                "ordered_batch_input_sha256"
            ] = selected_hashes

        def verify(self) -> SubsetCache:
            for index in INDICES:
                base.load_batch(index)
            return self

        def load_batch(self, ordinal: int) -> object:
            return base.load_batch(INDICES[ordinal])

    rows_all = json.load(open(CORPUS, encoding="utf-8"))
    rows = [rows_all[index] for index in INDICES]
    stage(
        "subset_selected",
        counts={
            split: sum(row["split"] == split for row in rows)
            for split in ("selection", "fit", "validation")
        },
    )
    donor_tokenizer = load_local_tokenizer(PARENT)
    target_tokenizer = load_local_tokenizer(TARGET_TOKENIZER)
    bridge = load_strict_tokenizer_bridge_artifact(
        TARGET_TOKENIZER,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    subset = SubsetCache()
    trace = _trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache_manifest=subset.manifest,
    )
    installability = Lfm2CpcfInstallabilityAdapter(subset)
    phases = build_lfm2_cpcf_target_phases(
        trace,
        installability.iter_compiler_inputs(),
    )
    stage("phases_materialized", phase_count=len(phases))

    device = torch.device("cuda:0")
    parent = load_local_lfm2_model(
        PARENT,
        device=device,
        dtype=torch.bfloat16,
    )
    parent.requires_grad_(False)
    parent.eval()
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="LiquidAI/LFM2.5-230M",
        parent_tokenizer_fingerprint=(
            fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        ),
        evidence_fingerprint=base.cache_key,
        child_name="neural-forge/lfm2-v4096-100x-cpcf-subset-preflight",
        compression_ratio="100",
        target_vocabulary_size=(
            bridge.transport_profile.target_row_capacity
        ),
        child_tokenizer_fingerprint=(
            fingerprint_hf_tokenizer_semantics(target_tokenizer)
        ),
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=32,
        maximum_candidates=8,
        selected_candidate_rank=None,
        blank_seed=0,
        device=device,
        dtype=torch.bfloat16,
    )
    target = blank.model
    target.requires_grad_(False)
    target.eval()
    stage(
        "blank_materialized",
        target_unique_parameters=blank.unique_parameters,
        hidden_size=int(target.config.hidden_size),
    )
    compiled = compile_lfm2_cpcf_model_fold(
        parent,
        target,
        phases=phases,
        source_tied_table=_mapped_source_tied_table(parent, bridge),
        blank_target_report=blank.as_dict(),
        planned_target_unique_parameters=blank.unique_parameters,
        policy=Lfm2CpcfModelFoldPolicy(
            ridge=1e-8,
            state_reconstruction_weight=0.05,
            minimum_predictive_energy_fraction=0.90,
            minimum_validation_block_retention=0.0,
            enforce_validation_block_floor=False,
            short_conv_gate_policy=Lfm2CpcfShortConvGatePolicy(
                enforce=False,
            ),
            promotion_compression_ratio=100.0,
            require_promotion_ratio=True,
        ),
    )
    stage(
        "success",
        result_sha256=compiled.result_sha256,
        layer_count=len(compiled.layer_results),
        parameter_receipt=dict(compiled.parameter_receipt),
        boundary_bundle_sha256=(
            compiled.boundary_bundle.report["bundle_sha256"]
        ),
    )


if __name__ == "__main__":
    main()
