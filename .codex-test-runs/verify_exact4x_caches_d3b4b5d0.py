from neural_forge.hf_lfm2_circuit_trace_cache import (
    open_lfm2_circuit_trace_cache,
)


PATHS = (
    "/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95",
    "/home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b",
)


for path in PATHS:
    cache = open_lfm2_circuit_trace_cache(path, verify=False)
    print(cache.cache_key, cache.implementation_fingerprint, cache.batch_count)
