# C10 D4 design sweep: the K3-union target (written 2026-10-11, before any sweep result existed)

## Why

The pre-1.5.0 C10 target, `Y = MLP_parent(x) + sum_l FFN_k3,l(T x + b) U^T`, collapsed the trunk in the
M4 rehearsal: the 0-26 hybrid agreed with the S3 trunk on 7 of 210 R-Q positions (KL 5.75; the residual
reached 9.0x the trunk norm at L8; the K3 term was ~78% of the target's energy). This sweep looks for a
target that keeps the trunk coherent while carrying as much K3 as the trunk tolerates. There is no
training: every variant is a fold (selection + ridge refit) with a different target.

## Variants (c10_union_install.py 1.5.0, `--sweep-targets`)

`Y = a * BASE(x) + w * TERM(x)`, with prior `[a * trunk_down | w * prior_k3]`. TERM is the charted
K3 sum. BASE is the S3 trunk's own FFN (`trunk`) or the parent MLP (`parent`).

| name | spec | a | w |
|---|---|---|---|
| legacy | parent/additive:1 | 1 | 1 (the pre-1.5.0 target, as a reference row) |
| a0 | trunk/additive:0 | 1 | 0 (control: should be the trunk) |
| a003 | trunk/additive:0.03 | 1 | 0.03 |
| a01 | trunk/additive:0.1 | 1 | 0.1 |
| a03 | trunk/additive:0.3 | 1 | 0.3 |
| b01 | trunk/convex:0.1 | 0.9 | 0.1 |
| b025 | trunk/convex:0.25 | 0.75 | 0.25 |
| b05 | trunk/convex:0.5 | 0.5 | 0.5 |
| g01 | trunk/rms:0.1 | 1 | 0.1 * \|\|BASE_held\|\| / \|\|TERM_held\|\| per layer |
| g03 | trunk/rms:0.3 | 1 | 0.3 * \|\|BASE_held\|\| / \|\|TERM_held\|\| per layer |

Fixed for every variant:
- K3 dense L1-92 (`k3-run-oct09-claude/fold/dense`) with c17's 92->64 alignment (dense L0 skipped).
- c17 ridge charts L00-L63 (merged receipt: 64 present, 0 invalid).
- `--k3-channels 12288`. This is at most 12,538, so the five media units stay budgeted.
- `--chart-bias-ceiling 0`, i.e. bias is recorded only.
- The 27B evidence manifest (`a0a7f0da...`; 18 fit / 6 held rows of 4096 tokens).

## Approximation

The sweep is open loop. Every trunk layer is fitted on the S3 trunk's own forward rows. A real install
is closed loop, where each layer is fitted on the rows of the child built so far. This is what lets all
ten targets share one capture, one pool Gram and one selection, and lets the 64 layers run as three
independent ranges on three GPUs.

The evaluation is honest about this: each variant is run as a real virtual child, so the
mismatch is measured, not assumed away. The `legacy` row is the calibration: it reruns the old target
open loop on all 64 layers, so it can be read against the closed-loop 0-26 hybrid's 7/210.

## Evaluation (c10_virtual_child_check.py 1.2.0, GPU6 bf16, one pass)

Measured per variant against the S3 trunk and the 27B parent (the S0 Chimera import at
`/mnt/ai_data/omni-campaign/s0-probe-sep04`; its streamed exactness receipt vs the HF parent reads
210/210 argmax):
- **Held rows** (6 x 4096, teacher forced): mean next-token NLL over the base vocab, top-1 agreement
  with the trunk and with the parent, KL(trunk || child).
- **The 8 R-Q prompts** (210 positions): top-1 agreement with the trunk and with the parent, KL.
- **Per layer**: residual relL2 vs the trunk and the residual norm ratio.
- **From the sweep records**: the K3 term energy fraction of the target (`||w TERM|| / ||Y||`, held)
  and the share the installed K3 columns write (`k3_columns_write_fraction_held`).

## Selection rule (pre-registered, a DESIGN choice and not a gate)

1. **Coherent** means all three hold:
   - R-Q top-1 agreement with the S3 trunk >= 0.80;
   - held mean NLL <= the S3 trunk's held NLL + 0.10 nats;
   - max over layers of the held residual norm ratio <= 1.5.
2. Among coherent variants, pick the largest mean (over the 64 layers) held K3 term energy fraction.
   If two are within 0.01 of each other, the lower held NLL wins.
3. If none is coherent, D4 stays open. The table is reported as is, and the variant closest to coherent
   is named.

The C10 default target is NOT changed by this sweep. The pick is a recommendation for the judge
(orchestrator / Cyril). The real install would then pass `--union-target <spec>` closed loop.
