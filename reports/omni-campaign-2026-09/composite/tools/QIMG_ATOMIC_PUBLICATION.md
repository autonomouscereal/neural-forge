# QIMG stack publication V5

The existing stack capture and stack consolidation producers publish complete,
flushed same-directory pending files without replacing existing artifacts.
Hardlink remains the first operation. Only Linux link errors EPERM, EOPNOTSUPP /
ENOTSUP and ENOSYS permit libc renameat2 with RENAME_NOREPLACE (flag 1).
No replacement, copy, target-existence emulation or guessed syscall number is
used. Other errors propagate unchanged. Missing syscall/libc/kernel/filesystem
support fails closed; the refused pending file remains available for review.

Root's actual 2026-10-06 Lexar probe established exfat link EPERM, renameat2
success and EEXIST refusal. The separately pinned exact-module probe also checks
the stage-inode guard and eight concurrent publishers before adoption. Tests
force the measured link limitation only; actual libc/kernel rename is exercised,
including a full tiny60 source/capture/120-fit/1933-key/native-forward route.
Actual original-checkpoint execution remains a separate resource job.

After rename succeeds the producer's pending file is consumed. Every producer
readback uses the destination. Failed publication retains the stage; a later
post-publication readback failure leaves the canonical output visible but never
accepted. These writers neither clean failed artifacts nor overwrite them.
The contract is atomic complete-file visibility and exclusion, not certification
against power loss. Producers flush/fsync file bytes before publication.

The new helper's raw source hash and exact version are included in capture and
fold implementation recipes and root policy metadata. Capture source pins change
explicitly: a V5 reader does not silently reinterpret V4 production captures or
resume them under a different generator. Original V4 snapshots/receipts and the
thirteen accepted single-block writer files remain byte-preserved. The original
single-block write_tensors path uses exclusive byte creation and has no hardlink
publication to retrofit. No finite fidelity or compression observation becomes
an acceptance/refusal gate. ImageEdit and behavioral/trained-model acceptance are
still outside this publication engineering result.

Entry point remains c16_qimg_stack_capture.py and c16_qimg_stack_fold.py, supplied
together with c16_qimg_atomic_publication.py from the exact new source packet.
The existing native stack file and seven native QIMG files are unchanged.
