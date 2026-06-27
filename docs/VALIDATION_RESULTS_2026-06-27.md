# 2026-06-27 25x Blank-Target Compression Validation

Donor: `Qwen/Qwen3-4B-Base`

Blank target shape/tokenizer: `HuggingFaceTB/SmolLM2-135M`

Folded child size: `162,826,560` parameters, roughly 25x smaller than the donor.

Remote run root: `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627`

## Best Current Checkpoint

`qwen4b_mean_cached_cka_mixed8192_1200s_best`

Pipeline:

1. Training-free donor fold into blank SmolLM2-135M shape.
2. Cross-tokenizer broad recovery with corrected projected-vocab masking and span CKA hidden transfer.
3. High-signal EOS recovery.
4. Mixed 8,192-row recovery using high-signal rows, arithmetic/choice rows, and clean curated 2026 v9 shard rows.

## Comparison

| Checkpoint | PPL mean | Built-in choice | Generation probes | ARC-Easy limit-20 | PIQA limit-20 | Manual notes |
|---|---:|---:|---:|---:|---:|---|
| Random blank SmolLM2-135M | 85894.18 | 0/4 | gibberish | not rerun | not rerun | no useful text |
| Raw 25x fold | 19840.45 | 1/4 | 0/3 | not rerun | not rerun | incoherent |
| Old broad xrecover | 5454.16 | 1/4 | 0/3 | not rerun | not rerun | repetitive |
| Old readable high-signal | 4239.15 | 1/4 | 3/3 | 0.30 | 0.50 | readable pruning/code/merge answers |
| P-KL + high-signal | 1677.10 | 1/4 | 1/3 | 0.40 | 0.50 | lower PPL but manual text degraded |
| CKA + high-signal | 3212.60 | 1/4 | 3/3 | 0.45 | 0.55 | readable, better than old high-signal |
| CKA + mixed8192 | 3104.34 | 4/4 | 3/3 | 0.50 | 0.65 | best automated score; still weak manual generalization |
| CKA + response-auto targeted | 2702.51 | 4/4 | 3/3 | 0.45 | 0.65 | best PPL; manual failures remain, ARC regresses |
| X-tokenizer response+behavior recovery | 2189.62 | 3/4 | 3/3 | 0.45 | 0.60 | better PPL, but behavior only 1/6 and manual text still collapses |
| Cross-tokenizer head calibration | 22103.76 | 2/4 | 2/3 | not run | not run | projected-logit fit improved, but generation degraded badly |
| Omnicoder curated recovery | 1754.09 | 4/4 | 3/3 | 0.40 | 0.60 | best PPL/choice after 25,874-row curated recovery; behavior still 1/6 |
| Curated + targeted behavior | 2808.69 | 3/4 | 3/3 | 0.40 | 0.50 | behavior improved to 3/6; arithmetic/code exactness still fail |
| Curated + targeted + EOS | 2865.05 | 2/4 | 2/3 | 0.45 | 0.50 | EOS stops runaway outputs; wrong numeric answers remain |
| Task alpha 0.60, curated->targeted EOS | 2170.67 | 3/4 | 3/3 | 0.40 | 0.45 | blends behavior to 3/6 with less PPL hit, but benchmark regresses |
| Donor Qwen3-4B slice | about 270 PPL in lightweight eval | 4/4 | useful | 0.80 | 0.80 | prior reference run |

All lm-eval values above use `--limit 20`; they are regression signals, not final benchmark claims.

## Manual Inspection

The best checkpoint answers these usefully:

- `Describe neuron merging in one sentence.`
- `Write a tiny Python function that adds two numbers.`
- The built-in pruning and merge-risk generation probes.

It still fails or degrades on:

- `Write a Python function multiply(a, b).`
- `What is 12 + 19? Answer only the number.`
- `A driver sees heavy rain. What should they do?`
- free-form tokenizer mismatch answers become verbose/repetitive after mixed recovery.

The response-only recovery option (`--response-loss-mode auto`) improved PPL from `3104.34` to `2702.51` and kept built-in choice/generation probes green, but it did not fix the manual failures above. Because ARC-Easy limit-20 dropped from `0.50` to `0.45`, the response-auto checkpoint is not the current best overall.

Later recovery ablations showed the same split:

- Curated Omnicoder recovery improved PPL to `1754.09` and preserved `4/4` built-in choice, but did not improve behavior-suite generalization.
- Targeted behavior recovery improved behavior-suite pass rate to `3/6`, but reduced ARC/PIQA and still failed held-out arithmetic (`12 * 13`, `19 + 26`) and exact named-code generation.
- `--append-eos` fixed answer/code runaway continuations, producing concise stops, but it did not fix numeric correctness.
- Same-shape task arithmetic from curated to targeted EOS found a compromise at alpha `0.60`, but it still regressed external benchmarks.

The strongest current evidence is that 25x blank-target folding transfers partial linguistic and domain behavior, but numeric/token formatting remains the hard failure. This points to needing numeric/token-aware cross-tokenizer alignment or blockwise activation reconstruction, not just more response-only SFT.

## Current Conclusion

The 25x blank-target fold is no longer just fitting tensor shapes. It transfers enough behavior to beat the random blank by a wide margin, generate some useful short text, pass the lightweight choice probe after mixed recovery, and improve ARC/PIQA slices versus earlier children.

It is not yet a successful replacement for donor capability. The remaining failure mode is narrow recovery and weak generalization: benchmark slices can improve while manual arithmetic, simple code variation, and common-sense instruction following remain unstable.

Next useful work should prioritize numeric/token-aware cross-tokenizer objectives, blockwise activation reconstruction, and representation-preserving fold repair over more one-off choice tuning.
