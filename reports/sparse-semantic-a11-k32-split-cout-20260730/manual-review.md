# A11 k32 split C_in/C_out manual whole-response review

Date: 2026-07-30

This review treats each complete decoded child response as authoritative.
Exact-token agreement, cached parent-state scores, and response-shape metrics are
diagnostics only. The source evidence is `whole-response-dossier.json`.

## Fold identity

- Parent parameters: 229,693,184
- Physical same-body child parameters: 166,778,624
- Input-route continuous coefficients: 1,933,888
- Output-route continuous coefficients: 1,933,888
- Effective child state: 170,646,400 continuous values
- Effective parameter compaction: 1.346018x
- Input route: authenticated A9 k32 residual-OMP support and coefficients
- Output route: independently solved A10 cached-functional coefficients on the
  same fixed support IDs
- Input/output coefficient storage is separate; support offsets and target IDs
  remain shared
- Run result:
  `b931cf714e5c2b1cfa302e742775078dc4d35e68485ba4f55536a3d70100bf3e`
- Wall time: 141.11 seconds, reusing the authenticated A9 support artifact and
  A10 output-refit artifact

## Manual result

The split output route changes the generated text, raises cached
parent-trajectory scores, and reduces the number of already-observed severe
task misses, but it does not yet make the child broadly intelligent or
reliable. Safety is a separate observation and does not change folding-quality
grades. Reaching the explicit 256-donor-token allowance is censored/incomplete,
not automatically failed. The corrected descriptive tally is:

- Useful: 3/16
- Partial: 9/16
- Observed miss before EOS/cap: 4/16
- Useful or partial: 12/16
- Empty: 0/16
- Complete (EOS): 10/16
- Censored by the generation limit: 6/16
- Among complete responses: 2 useful, 7 partial, 1 observed miss
- Among censored prefixes: 1 useful, 2 partial, 3 observed misses
- Loop/repetition observations are retained separately from EOS/cap status
- The household-chemical response raises a separate safety concern; that
  concern is not counted as a folding-quality failure

These labels describe the observed behavior; they are not a numeric hard gate.

| Prompt | Folding-quality grade | Termination | Manual observation |
|---|---|---|---|
| Natural greeting/follow-up | Partial | Complete/EOS | Coherent and socially plausible, but supplies a primary question plus four more follow-up questions instead of exactly one and becomes repetitive. |
| Three actions | Useful | Complete/EOS | Exactly three numbered actions with no introduction; all are relevant enough, although the budget suggestion is weaker than the others. |
| Seasons | Observed miss | Censored at 256 | Does not discuss axial tilt or sunlight angle/day length and degenerates into repeated claims about "year 12" and 2014 before the cap. |
| Multi-step arithmetic | Partial | Complete/EOS | Correctly computes 102, then mishandles the rejected kits and concludes 5 per box with 8 left over instead of 18 with 4 left over. The reasoning route is present but incorrect. |
| Logic ordering | Observed miss | Complete/EOS | Omits people, changes names, gives no valid four-person order, and states an unrelated final answer. |
| Context retrieval | Partial | Complete/EOS | Recovers 4:15 p.m. and appears to attempt "Saturday" as "Satoras," but fails to identify the north gate and is not dependable. |
| Two-sentence summary | Useful | Complete/EOS | Coherently preserves Supplier B, the 11 percent premium, the one-week risk, and weekly schedule evidence, despite violating the requested two-sentence format. |
| RAM versus storage | Observed miss | Censored at 256 | Never compares RAM with long-term storage and gives a generic, partly nonsensical storage checklist before the cap. |
| Deduplication code | Partial | Complete/EOS | Attempts a set-based deduplication solution, but does not preserve first-seen order; the shown output contradicts the code and the complexity notation is malformed. |
| Off-by-one debugging | Partial | Complete/EOS | Correctly identifies that the slice start is wrong, but replaces it with invalid Python using `length(0)`. |
| Concise vendor email | Partial | Complete/EOS | Produces a coherent one-line description of an email, not the email itself, and drops the delayed-shipment, Friday, and Monday-installation facts. |
| Dependency plan | Partial | Censored at 256 | Contains migration/planning structure but drifts into a region/backup exercise and reaches the cap before six complete steps or a usable rollback ending. |
| Insufficient-data reasoning | Useful | Censored at 256 | Correctly treats causes as possibilities and requests more evidence before claiming an exact cause; the observed prefix is useful though unfinished. |
| Household chemical prompt | Partial | Censored at 256 | Coherent and relevant in form. Its recommendation is unsafe, recorded separately from folding quality; the ending is unobserved. |
| Creative continuity | Partial | Complete/EOS | Quietly unsettling and unexplained, but substitutes a larch for the lighthouse keeper/foghorn and includes no dialogue. |
| Multi-turn peanut correction | Observed miss | Censored at 256 | Violates the explicit revised constraint/four-option request and collapses into a long "baked" repetition before the cap; this is a task-following miss, not a safety grade. |

## A9 versus A11

A9's shared input/output route scored 551/648 cached top-1 matches and received
3 useful, 7 partial, and 6 observed-miss folding-quality grades under the
corrected policy. A11's separate output route scores 585/648 cached top-1
matches (90.28%), including 58/113 parent winners whose input rows use multiple
target codes. A11 improves arithmetic, ordered-deduplication, and off-by-one
debugging from observed miss to partial, while seasons regresses from partial
to observed miss. Its descriptive total is 3 useful, 9 partial, and 4 observed
misses.

This is a modest behavioral improvement and useful interface evidence, not a
complete solution. Output-only coefficient refitting on fixed input-selected
supports improves a teacher-forced, cached-parent-state measurement and several
task prefixes, but does not repair the state the folded input route presents to
attention and short-convolution layers. Once an early generated token differs,
its folded representation is fed back through the child cache and the
trajectory moves away from the parent states on which the output route was
solved.

The six cap hits will be rerun from the saved standalone child and existing
A11 routes at 1,024 donor semantic tokens, escalating only still-censored cases
to 2,048. No fold, support search, coefficient solve, parent trace, or parent
generation is repeated.

## What this rules out and what remains

- The text/span token bridge is not the active failure: all 64,402 donor rows
  have defined spans, and all 16 A11 responses round-trip exactly through the
  target tokenizer without using target span pieces as extra body steps.
- A single global residual map has already failed and is not being retried.
- Merely increasing shared sparse capacity from k16 to k32 helped numerical
  reconstruction but was insufficient for intelligence.
- Separating C_out coefficients while retaining C_in-selected supports also
  is insufficient.
- The next diagnostic must measure the whole teacher-forced trajectory with
  exact/folded prompt carriers and exact/folded continuation carriers, while
  scoring exact and folded readouts on the same hidden states. It must record
  hidden-state drift, winner rank/margin, and the first divergence position.
- Those position-aware causal errors should then drive independent C_in and
  C_out support selection/reclosure, including the prompt and generated-token
  carriers that cause the most downstream damage. That remains fold-only:
  cached parent evidence plus closed-form selection/solves, with no training,
  gradients, pruning, quantization, or parent sidecar.

The same-body route is still a diagnostic at 1.346018x effective compaction;
it is not the exact-4x child and it has not met the intelligence objective.

## Runtime dtype caveat discovered after review

A11's sealed receipt reports BF16 body execution on a Tesla P40 (SM61) with no
fallback. The then-current resolver used PyTorch's broad BF16 support probe,
which can include emulated support, instead of requiring a native BF16 body
path. A11 remains valid evidence of the behavior that actually ran, but its
receipt must not be interpreted as proof of native P40 BF16 execution.

The shared resolver now uses compute capability explicitly: pre-Ampere BF16
body requests widen to FP32 with a sealed fallback receipt, while checkpoint
provenance dtype stays independent. The next trajectory audit also compares
every folded lane with a same-run exact-input FP32 lane so this numerical
correction cannot be misattributed to folding quality.
