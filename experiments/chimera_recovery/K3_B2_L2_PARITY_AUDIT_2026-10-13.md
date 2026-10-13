# K3 B2/L2 parity audit, 2026-10-13

**The real cause remains unresolved. No production arithmetic change is justified yet.** The supported next experiment is a shared-input L2 replay with whole-corpus and grouped FFN calls, identical parent weights, original runtime settings, repeat controls and intermediate tap hashes. This experiment is backlogged behind recovery; no GPU was allocated for this audit.

The original external K3 production caches were skipped after B2 failed its unchanged p99 gate (1.2499980926513672 > 1.0). That is separate from the existing OMNI1 child and its current recovery. D1 does not consume those skipped external K3 caches. This audit does not change OMNI1, the folding recipe, the original queue, existing source jobs, cache seals or parity thresholds.

## Verified evidence

The actual completed bisect used the fold's full 143,360-position padded corpus, identical source files/input IDs, row chunk 1, token chunk 1024, BF16 parent projection dtype and the same sm75 card class. All six lane source hashes and the fold-engine SHA match the original fold receipt. L0 and L1 teacher prefix hashes are identical. The first unequal prefix is L2:

- Original fold L2 teacher: `99eb1700d774e2805f25a886ec2cdfd907669053078876a31de39408168f664f`.
- Bisect L2: `6db46e44574f2f75a3319d83c13fdde936fba57c1992a0dd4f8114ff933ac673`.
- Completed bisect receipt: `a20d5d9a2b88ff269513286f1951a121545143783953aa339aa4dbb225ef3039`.
- Original full fold receipt: `8948bab7feeaa3cd460418f3552600b2d93a6c944143cfedb1b3a9122fa787df`.
- Exact original fold engine: `fd7dbf8da04504696ebb0702372802d2869bd1c13ee8c50d2578cee8108a55b7`.
- Exact frozen cache tool: `7899023cb8eb411c4b0fef005bb135970aaef65bb42d4653c688800bf7e88826`.

The original source digest receipt is verified, and the bisect recorded unchanged config/index/file metadata before/after with no drift versus the fold. Its inline full-shard digest was disabled. This audit downloaded/hash-checked only the pinned source/receipt files and the small parent config, and read exact snapshot headers/existence metadata; it did not rehash the full parent weights.

## Concrete execution difference

In the original `k3_dense_pass.py`, `DualStreamRun.run_layer` calls `attention_half` and `ffn_half` once over the entire 143,360-position corpus. `attention_half` itself uses the same row-chunked attention. In the frozen cache tool, `CachePass.layer` invokes those functions independently for five groups: **32,768 / 32,768 / 32,768 / 24,576 / 20,480 positions**.

The router FP32 `F.linear` and each selected-expert BF16 `w1`, `w3` and `w2` GEMM therefore receive different M dimensions. This is a concrete numerical execution difference; it is a plausible explanation for the first divergence, not proof of its cause. Raising `--group-tokens` alone cannot reproduce the original all-corpus FFN call: `TeacherCorpus` always keeps segment boundaries, so even a budget of 143,360 yields three groups (98,304 / 24,576 / 20,480). The original B2 also used a review-only corpus, which differs from the corpus that produced its reference distribution.

GroupView offsets and sequence lengths preserve complete rows; the supplied `--no-trim` keeps the original padded T. The in-place cache implementation pushes the entering prefix into the snapshot block before overwriting that group's prefix, matching the original residual-pair update. No source-level cropping, attention-mask or residual-stack error was found in this comparison.

L2 has an exact-zero `self_attention_res_norm` gain at coordinate 7064. Both the pinned harness and the captured original stock `_apply_attn_res` multiply norm and projection weights; neither divides by this gain. The captured parent config has `num_expert_group=topk_group=1`, so the stock grouped-topk mask branch is inactive. These facts do not establish full parent parity; they rule out the specific suspected zero-gain division and omitted active group-mask branches here.

The captured stock MoE gathers/sorts expert tokens and reduces selected outputs along the top-k axis; the harness accumulates selected experts in ascending expert order in FP32. This is an additional stock-versus-harness numerical ordering difference, shared by both historical fold and cache engine paths. It does not by itself explain the new fold-versus-cache L2 gap.

Runtime evidence also has limits: the probe did not record all backend reduction flags; the fold records 22 CPU threads while the probe records 16; Python patch versions differ. Matching source and card class is not an execution-equivalence certificate.

Stock capture paths, with current source identity kept separate from historical receipt pins:

- `/home/cereal/omni-campaign/k3/k3_src/modeling_kimi_linear.py`, SHA `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`.
- `/home/cereal/omni-campaign/k3/k3_src/configuration_kimi_k3.py`, SHA `735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae`.
- Actual parent `config.json`, matching historical SHA `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`.

## Implemented diagnostic and actual validation

`k3_l2_diagnostic.py` verifies the original source capsule and receipt pins before preparing a plan. Its `replay_shared_pair(engine, runner, pair, corpus, groups, device, row_chunk=1, token_chunk=1024)` API calls the genuine engine on the same immutable residual pair, records carried state / FFN input / latent input / router IDs and weights / routed accumulator / prefix output, verifies input preservation, and computes bounded quantitative comparisons. It never fits a child or declares parity/admission. Call it twice on the same loaded input/weights to obtain whole/grouped repeat controls.

Each tap includes a raw-byte seal and **the original FP32-upcast-byte digest convention** used by `k3_common.sha256_tensor`. Comparing a raw BF16 hash to a historical prefix hash would be invalid. The convention is checked against the authentic source on a multi-chunk signed-zero fixture.

The private CPU capsule exercised actual original KDA, AttnRes and packed-MXFP4 MoE operations on explicitly authored small weights, including a zero norm gain. **11 tests passed, zero skips**, 2.204 seconds; test and standalone control commands both exited 0. The owned operation finished in 5.010 seconds with CUDA hidden, four threads, 8 GiB address-space and 60-second CPU/deadline bounds. It loaded no real parent weights. Both layouts repeated bitwise and matched on these small fixtures; this does not reproduce or explain the real L2 discrepancy.

Actual immutable remote CPU evidence: `/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/k3-l2-cpu-control-v3/output/{execution,control}.json`.

- Execution SHA: `91ba74d2fa461f030431f09b2047ce7fdc44eb567eb4200c1e21795e3416862f`.
- Control SHA: `e4a77b684cace3d2bd63f9ad067c9b8e7b2be39ca8beb474f72bb3767dffbcad`.
- Prepared diagnostic plan v2 SHA: `8701723453d9a4e87720417be0b00c353a17ebf86147e3b5bab125c7815a5d44`.

## Next genuine experiment, after recovery resources permit

The original receipt records snapshots before L2/L3 and held L2 FFN tensors, but all three are absent at their exact host paths under `/home/cereal/omni-campaign/k3-run-oct09-claude/fold/`. Their references are not usable input artifacts. **An original teacher forward over L0 and L1 must regenerate the residual pair entering L2**, without fitting or replacing any child weights. Require the full original source corpus and both old prefix hashes before comparing L2.

Use one pinned original engine/parent and the exact full padded corpus. Run whole-corpus A/B and five-group A/B from the same regenerated teacher pair, preserve the complete block stack, and record source hashes, kernel/backend/runtime flags and the original FP32 digest convention. Require each arm to repeat and inspect the first unequal tap. Only if whole-corpus L2 matches the old fold hash and grouped L2 matches the old bisect hash does the evidence isolate grouping as the cause. Otherwise retain the unresolved classification and compare the first failing sub-operation under controlled dtype/backend settings.

Do not select seeds, loosen tolerances, omit padded positions, compare unrelated corpus shapes, or overwrite old receipts. Even an explained L2 gap is not a full B2 pass: the original full-stack parity gates must still be measured unchanged on genuine outputs. Any later shape-policy fix requires a separate candidate and cache namespace. It remains outside the current recovery optimizer path.

Portable preparation command (paths refer to a separately supplied pinned source capsule):

```text
python -B experiments/chimera_recovery/k3_l2_diagnostic.py prepare --sources <pinned-capsule> --probe <original-probe_receipt.json> --out <new-diagnostic-plan.json>
```

For the actual source CPU tests, set `CHIMERA_K3_DIAGNOSTIC_SOURCES=<pinned-capsule>` and run `python -B -m unittest tests.test_chimera_k3_l2_diagnostic -v`. These commands allocate no GPU and load no real parent weights.
