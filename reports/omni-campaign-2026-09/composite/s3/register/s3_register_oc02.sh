#!/bin/bash
# S3 trunk child registration, OC02-F-Q38-27BU-C2K4352 (lane E1, 2026-10-09).
#
# Same argv as the Oct-08 dry run (/mnt/lexar/omni-campaign/scratch-s3-dryrun-oct08/
# s3_register_dryrun.sh): id, stage, wave, tag, parent, directory, the 64 MLP
# components, the passed-through list (incl. the mtp drop), recipe and recipe
# command are unchanged.  Only two things differ:
#   * the tool is the LIVE $OC/ledger/oc_checkpoint.py (its register/clone guard
#     now admits /mnt/lexar; delete clause iv-c still refuses all of /mnt);
#   * --notes is rewritten: the Oct-08 text said "frozen review not run", which
#     stopped being true on 2026-10-08 17:54Z, and it did not cite the exactness
#     receipts.  Every number below names the file it was read from.
#
# Usage:  s3_register_oc02.sh            # dry run, writes nothing
#         s3_register_oc02.sh --confirm  # seals the register record + ledger row
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
export OC_ROOT=$OC
CHILD=/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07
PARENT=/home/cereal/models/Qwen3.8-27B-Uncensored
TOOL=$OC/ledger/oc_checkpoint.py
CONFIRM=()
if [ "${1:-}" = "--confirm" ]; then CONFIRM=(--confirm); fi
echo "=== register with tool: $TOOL sha256 $(sha256sum "$TOOL" | cut -c1-64)  confirm=${CONFIRM[*]:-no}  ($(date -u +%FT%TZ))"
cd $OC
ARGS=(register --checkpoint-id OC02-F-Q38-27BU-C2K4352 --stage fold --wave oc02 --tag C2K4352)
ARGS+=(--parent-code Q38-27BU --parent-checkpoint-id PARENT-Q38-27BU)
ARGS+=(--directory "$CHILD" --parent-directory "$PARENT" --parameter-count auto --parent-parameter-count auto)
# 64 x (3 x 5120 x 17408 = 267,386,880 parent -> 3 x 5120 x 4352 = 66,846,720 child), 0.25 each
for k in $(seq 0 63); do
  ARGS+=(--component "mlp.layer$(printf %02d "$k"):267386880:66846720:model.language_model.layers.$k.mlp.")
done
ARGS+=(--passed-through gdn_mixer --passed-through attention_mixer --passed-through interface --passed-through vision)
ARGS+=(--passed-through "mtp:dropped-not-folded:424699392-parameters:text_config.mtp_num_hidden_layers-1-to-0")
ARGS+=(--recipe "$CHILD/qf-build-receipt.json")
ARGS+=(--recipe-command "qf_stream27.py --streamed --parent /parent --evidence /inputs/evidence/omni-evidence-27b-v1.json --tag S3-C2K4352-sep07 --parent-code Q38-27BU --mlp-keep 4352 --mlp-group 3 --drop-mtp --device cuda:0 (GPU4, 22 threads, nf-s3 source fb913c82, run-output /run-output/S3-C2K4352-sep07)")
NOTES="Codex Sep07 S3 trunk build (qf_stream27.py --streamed, image local/omni-s3-fold:sep07-ba37662f, source tarball \$OC/runtime/s3-fold-sep07-ba37662f/nf-s3-fb913c82.tar; diff vs canonical composite/tools/qf_stream27.py NOT recorded). \
qf-build-receipt.json sha 5fcefd596ea03691956bd0d6f4a3a1eb27a716e436c293e9bd81bca8782e5031; child_tensors_sha256 a42be5d82b67791f3c5bc83980d9a5be3218e1e7ecde6fe0f79e1adbb3ca9518. \
MLP 17,408 -> 4,352 on all 64 layers (each component 0.25 of its parent). \
MTP DROPPED, NOT FOLDED: 424,699,392 params, named in passed_through and equal to fold_accounting.unexplained; it is not credited to any component. The receipt's parent 27,356,728,560 = header parent 27,781,427,952 minus mtp; net_ratio here is against the full header parent. \
Receipt parity {} empty; GGUF not run; config.json declares dtype float32 while the shards are bf16. \
Registered IN PLACE on /mnt/lexar (local USB exFAT SSD, not the CIFS share) under the 2026-10-09 REGISTRABLE_UNDER_PROTECTED narrowing of oc_checkpoint.py; delete clause (iv-c) still refuses every /mnt path, so this record can never authorise an rmtree. \
HF-vs-native exactness (teacher-forced, 8 bank-v1 prompts / 210 positions, receipt_q27_streamed 1.0.0, chimera tree ef1ffb9c, native export /mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native): \
\$OC/composite/s3/receipt_s3_streamed_exactness_cpu_fp32.json (sha 8f7ee85813f1bb59...) CPU fp32 argmax 208/210, max_abs_logit_diff 0.1224; \
\$OC/composite/s3/receipt_s3_streamed_exactness.json (sha 3c9eab110fb1b5d6...) GPU bf16 argmax 201/210, max_abs_logit_diff 0.5312. \
Frozen review: \$OC/composite/s3/review/oc-dossier-OC02-C2K4352-NATIVE-NOTHINK.json (file sha 5fe7463cc258d651...) was decoded on the NATIVE export (runner run_omni_frozen_review 1.1.0, split 28/36, bf16, thinking off, cap 1024), NOT on this HF directory; the two are tied only by the exactness receipts above. Assessment \$OC/calibration/out/assessment-OC02-C2K4352-NATIVE-NOTHINK.json (sha fb3cd4fe75c3f398...): form boxes ECHO 8 / CAP_RIDE 5 / EMPTY 2 / CONTENT 1 of 16 (13 censored at cap, 5 loops), column A 8 / column B 1 of 15, both UPPER BOUNDS, unread by Cyril. No Q38-27BU parent surface exists on any runner."
ARGS+=(--notes "$NOTES")
timeout 280 $PY "$TOOL" "${ARGS[@]}" "${CONFIRM[@]}"
echo "=== exit $? ($(date -u +%FT%TZ))"
