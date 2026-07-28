# A0024 exact-4x baseline manual review

Date: 2026-07-27

## Proven structural result

- Parent: 229,693,184 unique dense parameters.
- Child: 57,422,440 unique dense parameters.
- Physical parameter compaction: 4.000059628256828x.
- Child topology: 14 native LFM2 layers, hidden size 776, intermediate size 700, Q10/KV5, head dimension 64, 4,096-token bridged vocabulary, untied input/output tables.
- Storage: FP32 dense parameters. This is parameter-count compaction, not quantization.
- The complete child was saved, freshly reloaded, and decoded without a parent sidecar or shared parent parameters.
- Operation audit: zero training, optimizer, backward/autograd, pruning, quantization, sparsification, parent forward/generate, or parent trace recapture operations.
- Complete child state SHA-256: `b327007ca59962acf24694798dc9817eb3084a9dd0d6b6c0a36ab7920f097801`.
- Dossier file SHA-256: `40cf6bbdc14092b1df65d396f8e26af25688f3187a81f727e13018943679e841`.
- Canonical dossier SHA-256: `fcb3d09b37a2a8c2220c513097e5f4c87c0d45415dd72f759ac2ca915d70cbad`.

## Manual intelligence verdict

I read all 16 complete child responses in `whole-response-dossier.json`. The exact-token diagnostics were not used as an acceptance gate.

The child is not coherent or useful. Every response begins with unrelated numeric/code-like fragments and then collapses predominantly into repeated `0` tokens. All 16 responses consume the full 256-token allowance and none emits EOS. No response answers the requested task, follows its requested format, or preserves useful parent behavior.

| Prompt category | Manual verdict |
|---|---|
| Natural conversation | Incoherent; no relevant response or follow-up question |
| Three-action instruction | Numeric fragment followed by repeated zeros |
| Seasons explanation | Incoherent code/numeric fragments; no explanation |
| Multi-step arithmetic | No calculation or answer |
| Logic ordering | No ordering or reasoning |
| Supplied-context retrieval | Does not retrieve any requested fact |
| Two-sentence summarization | Does not summarize the decision or risk |
| RAM/storage comparison | No analogy or comparison |
| Python deduplication | No valid function, example, or complexity explanation |
| Off-by-one debugging | No diagnosis or corrected implementation |
| Concise vendor email | No usable email |
| Dependency-aware migration plan | No usable plan |
| Epistemic uncertainty | No safe conclusion or requested missing information |
| Household chemical safety | No safety guidance |
| Creative continuity | No coherent scene |
| Multi-turn allergy correction | Ignores the correction and supplies no snack plan |

Manual result: 0/16 coherent and 0/16 useful. The count is descriptive evidence, not a numeric promotion threshold.

## Interpretation and next controlled test

A0024 proves that the corrected recipe can construct, persist, reload, and execute a complete independent dense child at real 4x parameter compaction. It does not prove retained intelligence.

The behavioral failure remains consistent with an evolving-child causal-coordinate gap: the baseline installs folded components from cached parent evidence without recapturing the full current child prefix after every material installation. A0025 holds the target topology, evidence, fold recipe, precision, prompts, and parameter count fixed while enabling full-depth/full-prefix live-child recapture after each install. Its decoded responses will determine whether causal closure repairs this topology or whether the next full-child run should use the dynamically surfaced state/QK-protected topology.
