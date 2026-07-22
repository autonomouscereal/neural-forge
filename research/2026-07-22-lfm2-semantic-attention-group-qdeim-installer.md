# Coupled semantic attention group-QDEIM installer

Date: 2026-07-22

## Result

The sealed-cache intact-group attention Q-DEIM candidate now has an isolated
transactional publication seam. It compiles against cloned metadata-only
templates from one exact-zero target attention block, authenticates the
candidate and direct-parent Q/K RMSNorm lineage, then installs ordinary dense
Q/K/V/O and q_layernorm/k_layernorm Parameters in one six-tensor transaction.

This is the missing composition boundary between the cache adapter and the
layer-stack recipe. It prevents a coupled attention candidate from being split
back into independent QK and VO stages that could overwrite each other.

## Flow

1. Read and authenticate the live target's six distinct ordinary Parameters.
   Distinct Python objects are not sufficient: their byte storage spans must
   also be disjoint from each other and from every other model Parameter.
2. Prove all six target tensors are exact zero and capture model identity,
   topology, element count, Parameter identities, storage pointers,
   requires-grad state, gradient identities, and gradient tensor hashes.
3. Authenticate direct-parent q_layernorm/k_layernorm tensors against a sealed
   norm inventory that is itself bound to the parent identity, layer index, and
   dense source Q/K/V/O inventory.
4. Pass detached clones of blank Q/K/V/O only as shape/dtype templates to the
   sealed-cache compiler.
5. Return a typed evidence deficit without calling the installation
   transaction. No incumbent, random, structural-proxy, or legacy split-lane
   fallback is admitted.
6. For a dense candidate, verify the adapter and standalone compiler seals,
   actual candidate tensor hashes, source/cache lineage, target-template hash,
   exact public compiler schema/method, source state, live decoder/chart
   tensors, GQA topology and selected whole groups, POD/interpolation tensor
   hashes, prohibited-operation audit, intact-producer policy, and
   whole-response evaluation policy.
7. Freeze a prepared object bound to the exact live blank target.
8. Reauthenticate that target immediately before publication and atomically
   install Q/K/V/O plus the two direct-parent norm tensors through the existing
   rollback-proven six-Parameter transaction.

## Why the norm tensors are part of the transaction

An exact-zero blank child also has exact-zero per-head Q/K RMSNorm gains. A
correct Q/K projection planted behind zero gains is behaviorally inert. The
parent norm tensors are only one head-dimension vector each, so preserving them
costs essentially nothing while removing that cold-start dead path. They are
copied directly; they are not averaged, fitted, or selected by a score.

## Failure semantics

- Missing or rank-deficient true attention context remains a typed no-candidate
  result and never mutates the target.
- Stale source norms, candidate reports, source inventories, target state,
  topology, Parameter identities, storage, or gradients fail closed.
- Mutable CPU backup tensors are rehashed against their captured receipts
  before preparation verification, stale-target comparison, or rollback; a
  caller cannot alter a rollback payload while retaining its snapshot seal.
- A partial copy or post-copy integrity failure restores all six tensor values
  exactly and proves the original model fingerprint, identities, storage, and
  gradients before propagating the failure.
- Wrapper-level postconditions are inside the rollback boundary. Even if the
  underlying six-tensor copy succeeds and its returned mutation inventory is
  then inconsistent, the installer restores the exact blank snapshot before
  propagating the validation error.
- A cache compiler touching the target is treated as a contract violation and
  is rolled back before failure.

These are structural and authentication checks, not semantic-quality gates.

## Intelligence evaluation policy

Numeric metrics remain useful diagnostics for locating failure and comparing
experiments. They cannot automatically accept, reject, roll back, or stop this
candidate. A binary manual score threshold also has no iteration veto. The
primary evidence is the child's complete decoded behavior across broad prompts;
retention targets are objectives, not brittle cutoffs.

## Verification

Focused tests cover:

- exact publication of all six Parameters;
- direct-parent Q/K norm planting from an all-zero blank target;
- typed deficit with zero fallback and zero mutation;
- stale norm inventory rejected before compilation;
- stale live Parameter identity rejected before installation;
- injected partial-copy failure with exact six-tensor rollback; and
- aliased topology rejected before compilation;
- distinct Parameters with overlapping storage rejected before compilation;
- mutable snapshot-backup drift rejected before installation;
- wrong standalone compiler schema and mutated decoder payload rejected; and
- injected post-transaction validation failure with exact six-tensor rollback.

The focused suite passes 11/11. The existing cache-adapter and attention
transaction suites are also run together before integration into a body recipe.
