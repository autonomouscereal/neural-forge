# A0025 exact-4x live-child-prefix manual review

## Outcome

The complete 57,422,440-parameter child is not coherent or useful. I read all
16 decoded responses in full. None answered its prompt, followed its requested
format, used supplied context, completed the reasoning task, produced working
code, or gave a safe response. This is a descriptive manual observation, not
an automatic metric gate or an exact-token comparison.

Every response consisted primarily of arithmetic/code-like fragments,
repeated function words, punctuation, numbers, or long whitespace/repetition.
All 16 continued to the 256-token generation limit without EOS. The same
failure was visible in conversation, instruction following, explanation,
arithmetic, logic, context use, summarization, comparison, coding, debugging,
email writing, planning, uncertainty handling, safety, creative writing, and
multi-turn correction.

## What changed relative to the corrected A0024 baseline

Live full-depth child-prefix replay made several responses more lexically
varied and reduced the baseline's near-universal long suffix of repeated zero
tokens. It did not recover language, prompt understanding, or any usable
capability. Therefore stale causal coordinates were a real defect worth
correcting, but they were not the sole cause of collapse.

The next full-child experiment must change the automatically selected capacity
allocation, especially the joint ShortConv/state, Q/K, and MLP bottleneck. It
must not repeat H776 with another scalar alpha/ridge sweep. Component-level
measurements remain diagnostic inputs to the whole-model allocator; the next
artifact remains a complete independent exact-4x child.

## Artifact authentication

- Runtime commit: `92b2e091`
- Run: `fp32-source-local-residual-rho025-live-full-prefix-whole4x-a0025`
- Parent parameters: `229,693,184`
- Child parameters: `57,422,440`
- Physical parameter compaction: approximately `4.00006x`
- Child model-state SHA-256: `3f47d738b1c50a21d91d8c80fd9df310582bd7bac5c713455bbe6b804b8a85fc`
- Checkpoint inventory SHA-256: `524d7e7de4045a4d3cb0b1eb35e52151d80f8895d28b92da969906188329c9dd`
- Dossier file SHA-256: `b399ed8acaeabbeb34c8f09f77fa1a02513bfe32767397c6acecde014899cae7`
- Embedded dossier SHA-256: `c8ffb7622f1490034156f14f07ed7e78962319cf03a794184b3f1e65f9e5074c`
- Embedded dossier digest independently recomputed: exact match
- Backward calls: `0`
- Optimizer steps: `0`
- Training steps: `0`
- Parent forward passes during the fold/review transaction: `0`
- Failure receipt: absent; the run completed normally

The checkpoint was saved, freshly reloaded, and decoded before this review.
