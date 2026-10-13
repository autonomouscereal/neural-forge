D4 executable core, 2026-10-13
================================

`train_consolidation.py` supplies actual tensor/runtime components. It does not
launch a production model, publish a merged checkpoint or claim that D4 is ready.
No accepted D2 bank exists yet, and the full union constructor/cache dispatcher,
rolling safety/exact resume and native checkpoint writer are still required.

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
scatter-before-cast forward. Unit and music master activation remains a separate
union constructor obligation, not a claim inferred from interface activation.

The metadata stage chain now includes D3d_audio_ce after D3d, before D4. Its
authored audio CE is distinct from H3 generation and from future speech output;
no new replay quota or speech capability is invented here.

Meaningful CPU tests use the genuine campaign HybridD3 tiny source: joint runtime
versus numerical merge on text/ACE/H3/OCR, streamed gradients against a resident
full autograd graph, a second resident backward, exact native interface promotion,
observable double application, actual physical-owner aliases and composed-source
refusal. These are fixture proofs, not model intelligence or native media quality.

    CHIMERA_D3_HYBRID=/home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py python -B -m pytest tests/test_chimera_train_consolidation.py tests/test_chimera_recovery_consolidation.py -q

Local Windows has no Torch: 22 tests passed and3 genuine runtime tests skipped.
Root must execute those3 tests on the actual CPU runtime before merging/promoting
the numerical/runtime core. Source mutation/target admission and production
parity on the accepted D2/D3 lineage remain separate checks.
