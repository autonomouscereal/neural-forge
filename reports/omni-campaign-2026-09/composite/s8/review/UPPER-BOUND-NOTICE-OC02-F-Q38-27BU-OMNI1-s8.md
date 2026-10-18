# UPPER-BOUND NOTICE -- OC02-F-Q38-27BU-OMNI1 (lane S8-TEXT, 2026-10-13T08:55:46Z)

Every 'sealed' number for this checkpoint in the ledger, SCORECARD.md and REVIEW_QUEUE.md is the MACHINE
CEILING from oc_review_upper_bound.py over an assessment run WITHOUT --per-case. Nobody has read this
checkpoint's text. The parent's count is the parent's own ceiling (column A 15 / B 12), measured (E2).
canonical_content_miss is 0 by construction, so nothing in this package can make the checkpoint bad.

## Surface

- oc-dossier-OC02-OMNI1-NATIVE-NOTHINK.json: bf16, Quadro RTX 8000 (sm75, head) + 2x RTX 3090 (sm86), split 6/33/30 (GPU6 head + blocks 0-5; GPU4 blocks 6-38 incl. the 5 charted blocks 32-36; GPU0 blocks 39-68), torch 2.11.0+cu128
- run_omni_frozen_review 1.2.0 + run_review_finite_guard 1.2.0, chimera tree ef1ffb9c, bank v1 0ee6bf14..., greedy, thinking off, cap 1024
- --ledger-band-logit-mask ON (OMNI1's C18 music band rows win the argmax unmasked: 146/210 R-Q positions)
- parent: oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json (fp16, 3x P40); S3: oc-dossier-OC02-C2K4352-NATIVE-NOTHINK.json (bf16, 2x RTX 3090)

## To read

SCORING-SHEET-OC02-OMNI1-NATIVE-NOTHINK-vs-PARENT-s8.md carries this checkpoint's verbatim text with the parent's response beside each lane.
