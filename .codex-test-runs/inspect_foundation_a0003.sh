#!/usr/bin/env bash
set -euo pipefail

RUN=/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0003
sha256sum "$RUN/foundation-receipt.json"
jq -c '{schema,status,receipt_sha256,artifact_sha256,model_state_sha256,target_unique_parameters,operation_audit,legacy_cpcf_layer_compiler_executed,source_selected_tied_table_sha256}' "$RUN/foundation-receipt.json"
jq -c '{architectures,hidden_size,intermediate_size,num_hidden_layers,num_attention_heads,num_key_value_heads,vocab_size,tie_word_embeddings}' "$RUN/checkpoint/config.json"
