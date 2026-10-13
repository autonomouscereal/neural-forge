D4 executable core, 2026-10-13
================================

`train_consolidation.py` supplies actual tensor/runtime components. It does not
launch a production model, publish a merged checkpoint or claim that D4 is ready.
No accepted D2 bank exists yet. The trusted cache/objective dispatcher, exact
replay amendment selection, live allocator assignment, measured host memory
budget and registered native checkpoint writer remain required.

The existing HybridD3 source is an immutable D2 child or D2 compose plan. D2
r64/r32/gain are already included there. Reattaching that full bank would apply
it twice; freezing it would not train all D2+D3 owners. The new joint bank instead
requires authenticated original pre-D2 text weights and a complete D2 checkpoint.
`verify_d2_source` verifies the proposed binding schema, exact source identity,
r64/r32/gain site plan, actual COMPLETE.json and every producer file, source/data
identity, completed step, full adapter shapes/finiteness before copying. An
actual accepted-stage binding must be authored from the future completed D2 run;
absence is a refusal. It imports no D2 Adam state into D4.

`JointAdapterBank` retains distinct D2 and shared-D3 leaves, combines their low
rank factors at the actual trainer sites, and uses only D2's K3 adapter/gain.
`install_joint_bank` refreshes the bound factors on each streamed/resident layer
use. This prevents resident mode from reusing a freed concatenation graph on a
second backward. `fresh_union_optimizer` gives one physical leaf one owner in
one fresh AdamW, validates the nine named D2/D3 families by default, and scales
each pinned previous base learning rate by0.3. Different previous stage terminal
cosine rates cannot silently substitute for their base rates.

`merged_site_weights` computes both banks in fp32 on original per-site weights,
then folds the K3 gain into down columns exactly once and casts each completed
matrix once. It returns fresh tensors and never writes a model. It addresses
real qwen3_5 layer sites; converting those into exact Chimera fused tensor slices
and the ledger recipe is still a source-bound checkpoint writer obligation.

`activate_native_interfaces` converts inherited cumulative fp32 bridge/emission
masters into one compact native SliceDelta owner for each tensor. It removes the
old frozen overlay after copying the complete total, and retains the genuine
scatter-before-cast forward.

`consolidation_runtime.py` adds complete unit/music activation and a genuine
union constructor. It re-reads terminal native/shared exports from the same
producing checkpoint, proves every actual D2 text tensor against original+D2
composition, then reopens only the text engine on the original source. Native
reader and cumulative state remain bound to the D2 source. It rejects changed
weights, prefetch/layer use and incomplete unit/music masters. Only active code
rows are host fp32 optimizer leaves; ledger and reserve rows remain frozen.

The chain is D3d -> D3d_audio_ce -> D3d_speech_flow -> D4. Authored audio CE and
supervised continuous speech flow remain distinct objectives. Original replay
quotas stay unchanged here; the selectable speech amendment adds one replay per
accepted CE source row and one per accepted speaking source/sigma pair. Their
exact manifests/counts still need to be selected and bound by the dispatcher.

`UnionEngine` performs an actual fresh-Adam optimizer step over pinned admitted
train rows, in one exclusive ordered pass. Each objective uses the whole
accumulation window's measured supervised denominator and its declared weight.
It restores RNG and leaves row/step progress unchanged on nonfinite loss/gradient.
Saved full answers/outcomes remain the held-out evaluator's obligation, separate
from this optimizer. Checkpoints seal actual trainables, Adam state, per-owner
update counts, cursor, row order, source files and helper/runtime hashes; exact
resume preserves physical leaves and refuses mismatches. Dense host music Adam
is explicit candidate behavior and needs a host-budget check before production
selection. This is not a claim of sparse D3c optimizer parity.

Meaningful CPU tests use the genuine campaign HybridD3 tiny source: joint runtime
versus numerical merge on text/ACE/H3/OCR, streamed gradients against a resident
full autograd graph, a second resident backward, exact native interface promotion,
observable double application, actual physical-owner aliases and composed-source
refusal. These are fixture proofs, not model intelligence or native media quality.

    CHIMERA_D3_HYBRID=/home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py python -B -m pytest tests/test_chimera_train_consolidation.py tests/test_chimera_recovery_consolidation.py -q

The preceding numerical/interface core passed root's actual CPU run:67 tests,
1 expected missing-environment skip,10.47 seconds, unchanged staged source hashes
(recovery-preparation/d4_speech_cpu_1951_v3_result.json, actual19:44UTC). The new
union runtime has local23 passes and5 expected no-Torch skips across its combined
suite. Root must run the new genuine tiny replay/resume tests on the actual CPU
runtime before promotion. Source admission and accepted-lineage production parity
remain separate checks; fixtures never qualify as accepted D2/D3 exports.
