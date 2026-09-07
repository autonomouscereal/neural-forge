# Codex / Claude composite review — September 6, 2026

Work performed September 7 UTC. Continue the current composite campaign, not the historical LFM matrix. Claude's new K3, streamed-parity, chart-fit/install and frozen-review tools are substantial progress. They do not yet establish a verified all-source model, 27B forward parity, or deployment at 24 GB / 256k context.

## Completed narrow correction

`composite/tools/receipt_q27_streamed.py` now preserves the original FP32 GQA rotary inverse-frequency buffer before casting weights to BF16/FP16. The nonpersistent table cannot be repaired by checkpoint loading or by widening rounded values. This matches Chimera loader commit `75fabc5` and survives streamed block reuse.

- Before correction: 4 failed, 8 passed (12 focused CPU cases).
- After correction: 12 passed in 5.96 seconds.
- Independent reviewer: the same 12 cases passed in memory; no actionable findings.
- Full local suite: **20 passed, 2 failed, 5 skipped** in 171.83 seconds. Resident/streamed tiny-model mechanics passed. This is not a GPU or real-27B parity receipt.
- Remaining full-suite failures are in unchanged Windows paths: text-mode newline conversion makes the receipt's byte hash disagree with its sidecar; unavailable host RSS measurements are treated as zero and pass the budget check. These require separate correction, not a claim of a green full suite.

Runtime: Python 3.11.9, PyTorch 2.13.0+cpu. Focused command from neural-forge, with `CHIMERA_OMNI_SRC` set to the local Chimera repository and `PYTHONDONTWRITEBYTECODE=1`:

```powershell
& .venv/Scripts/python.exe -m pytest -o addopts='' -p no:cacheprovider reports/omni-campaign-2026-09/composite/tools/test_receipt_q27_streamed.py -k streamed_rope -q
```

Reviewed source SHA256: `c7fb1b0245d556c3af304ca6aeb9d78d9c294d39c682bcf93576e43488017257`.
Reviewed test SHA256: `bdc6e4c286569876a9818c1aba62f498ebba04c818c4a7202ae12466440ac96e`.

## Confirmed K3 resume defect — routed to Claude

`reproduce_k3_stale_prefix.py` runs a tiny three-layer CPU fixture: full pass at K64/pool1; fresh upstream pass 0–1 at K64/pool2; resume layer 2 at pool1. Layer 1's fold changes and layer 2's actual folded FFN input changes (relative L2 **0.240141337**, max absolute difference **1.245606573**). Teacher inputs remain bit-identical, but layer 2 still reports `fit_skipped` and reuses the old fold.

The compact `k3_stale_prefix_proof.json` was rerun after Claude's concurrent update. Production SHA was stable before/after at `a98041dfb3fe4edc592e99489121f9d2dc6de137bc30035c0e6134f58e362372`. The reproduction creates and removes its own temporary fixtures; no production K3 file was changed by Codex.

Suggested acceptance gate: stamp the upstream installed-fold lineage when fitting each layer; compare it with the current installed prefix before reusing that layer's done marker. Preserve normal same-prefix resume behavior and teacher-stream independence.

## Remaining coordination / course gates

- Claude retains active media work and GPU 5/6 reservations. No Codex GPU jobs, S0 re-import, training, remote writes, or server deletion occurred in this continuation.
- Pin the next real 27B parity receipt to the corrected streamed tool and verified Chimera tree; the prior S0 serialization verification is not forward parity.
- A source-level media precision concern was sent to Claude: charted forward narrows noise levels to the hidden dtype before timestep embedding. Test BF16 sigma 0.3/0.7; FP32 and exactly representable sigma 0.5 do not cover this concern. Not independently proven as a model-output defect here.
- H3 coreset evidence is not yet an installable H3 block or verified complete media model. Keep installation and whole-behavior gates separate from local fitting metrics.
- Relay project channel `chimera-omni` received the split, findings and proof. No attended Claude acknowledgement has yet been observed; do not describe the handoff as accepted.

## Git and retention

Original streamed files were saved before edits in commit `85a0881e35147720991ab7920598762e1c69f3e8`, on `codex/sep06-streamed-rope`. A separate index preserves the user's active HEAD and staged work. Use the final focused correction commit/diff, not a wholesale merge of the snapshot branch (its ancestry contains the earlier campaign safety snapshot).

Retain these three compact review files plus the source/tests in Git. Only this run's enumerated `pytest-before`, `pytest-after`, `pytest-suite` fixture directories and temporary `safety.index` are cleanup candidates; no existing model, campaign result, cache, or server path is included.

The environment blocked the scoped cleanup command before execution. Those temporary paths remain (19 test-fixture files totaling 21,090,367 bytes, plus the alternate Git index); no deletion occurred. Do not report them as removed. Correction/evidence commit: `37b59942de190dae97253fd11758ebedf2957db9`.
