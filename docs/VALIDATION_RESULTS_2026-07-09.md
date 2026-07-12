# Zero-Recovery Qwen 4B to Blank 158M Validation - 2026-07-09

## Contract

- Donor: `Qwen/Qwen3-4B-Base`, 4,022,468,096 parameters.
- Child: seeded random-init Qwen3, 158,288,256 parameters.
- Compression: 25.4123x.
- Recovery training, optimizer steps, and backward calls: zero.
- Parent evaluation was cached once and reused.
- Every completed lane ran 1,000 behavior probes, 1,000 exact-generation
  probes, 250 parent-choice probes, perplexity, and seven manual generations.

The cached parent scored PPL 877.5797, behavior 400/1000, exact generation
388/1000, and choice 250/250.

## Results

| Lane | PPL | Behavior | Generation | Parent choice | Manual verdict |
|---|---:|---:|---:|---:|---|
| Observable path-gated baseline | 1,459,654 | 0/1000 | 0/1000 | 154/250 | Incoherent |
| Observable path-gated plus one child-rollout refold round | 245,380 | 0/1000 | 0/1000 | 76/250 | Incoherent |
| Legacy global baseline | 296,982,020 | 0/1000 | 0/1000 | 146/250 | Incoherent |
| Legacy global plus one child-rollout refold round | 485,165,195 | 0/1000 | 0/1000 | 88/250 | Repetitive/incoherent |

The path-gated rollout lane accepted zero pair transactions and rejected its
final-interface transaction. Its internal rollout score therefore did not
improve: parent top-1 agreement remained 0.274171 and mean parent action
log-probability remained -6.731459.

The legacy-global rollout lane accepted one pair transaction. Its internal
metric appeared to improve from 0.307042 to 0.313802 parent top-1 agreement,
but mean parent action log-probability worsened from -4.131826 to -6.200757.
The external choice score then fell to 88/250 and PPL rose to 485 million.

All fourteen manual outputs were inspected. None answered basic arithmetic,
produced a usable Python addition function, explained tokenizer mismatch or
model merging, or gave a coherent lightning-safety answer. The outputs were
token fragments, multilingual noise, or repeated generic words.

## Decision

Child-action-only rollout scoring is rejected as a promotion gate. It asks the
parent whether the child's existing actions are plausible, which can reward
generic or repetitive actions without transferring the parent's useful action
paths. It is not allowed into the progressive proof.

The replacement gate uses fixed parent-generated anchor trajectories. Pair
fits use the union of current child trajectories and exact parent anchors, with
both trajectories for a prompt assigned to the same train/validation side. A
candidate must improve or preserve the child score on fixed parent actions and
must also pass regenerated child-rollout compatibility and degeneration gates.

The next compaction lane is progressive and activation-planned:

1. Qwen3 4B to a blank approximately 1.4B Qwen3 child.
2. Folded 1.4B child to a blank approximately 600M Qwen3 child.
3. Folded 600M child to the blank 158M Qwen3 child.

Each stage uses a globally optimized exact contiguous singleton/pair cover so
critical donor layers remain singleton whenever the constraints permit.

## Evidence

- Path-gated rollout archive SHA-256:
  `a1ac954f17ed86de087009f922a37a88fcb6c6223ebd2e423e1b3a7596c9125a`
- Legacy-global rollout archive SHA-256:
  `3e6c6497bae5b73259cd23c008045a875a643232afc2855ba536caa1792e7046`
- Local extracted evidence:
  `C:/Users/cereal/AppData/Local/Temp/qwen_rollout_refold_evidence_20260709`
