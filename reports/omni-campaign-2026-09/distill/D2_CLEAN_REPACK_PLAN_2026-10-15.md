# D2 clean re-pack: plan, manifest and cache job, 2026-10-15

**Written:** 2026-10-15, 02:20Z-04:10Z, by Claude, lane d2clean. "EST" marks an estimate. All times are UTC; host logs print UTC-6 and I converted them.

**Decision:** Claude-lead decision `d2-clean-repack-2026-10-15` (recorded in `STAGE_D_RULINGS_2026-10-12.md`). It upgrades the span mask of `d2-codex-heldout-exclusion-2026-10-15`. Not Cyril's words.

**What I changed on the server:**
- Added a new corpus version, `$OC/distill/corpus/text_clean_v1/`. The sealed `distill/corpus/text` and its 20 Q38 caches were not touched.
- Added two tools and their tests to `$T`.
- Installed a frozen copy of the sealed caches' tool, `$Q/frozen/qtext_teacher_cache-1.0.5.py`.
- Wrote `$Q/d2clean.go`.
- Started three runners (GPU3 now, GPU1/2 time-gated).
- Ran two probes of up to 12 minutes each on GPU3, under a `p40.claims` line and `gpu3.lock`.

Nothing of v1 was stopped or edited: the K3 identity gate, D1, the seal and post-D1 queues, the fold watchdog and the tool-call self-gen all kept running.

---

## 0. Bottom line

1. **The clean corpus exists and is verified.** It is `text_clean_v1`: manifest_sha256 `5f25c817fa06eb0030e32db6c88bfd50ee0233b9a2e6073cb3db096378cb8d52`, `verify` OK, post-pack review screen 0 hits in 3,015 rows.
   - No held-out token is an input or a target anywhere in it.
   - Every non-held-out document is trained exactly once, whole.
   - It carries 16,575,433 KD positions. The span-mask path's figure is computed by a different rule (section 5), so the two are not directly comparable.
2. **Only the new rows need a teacher cache.** That is 12 files, 12,349,164 tokens. The 1,497 kept rows reuse their sealed caches byte for byte.
3. **The P40s can build the cache.** They run the frozen 1.0.5 with the sealed flags except `--linear-attn-kernel torch`, because fla's Triton kernels do not compile for sm_61.
   - Measured rate: 53.21 tok/s per P40 at micro-batch 2.
   - Against an fp32 reference, a P40 cache is as faithful to the teacher as the sealed 3090 cache (section 3).
4. **The job is queued and running.**
   - The GPU3 runner is building `text-clean-00000`.
   - The GPU1/GPU2 runners wait for the tool-call refill's declared stop (21:40Z) and its claim release.
   - Every runner yields when the D1 bank v1 review wants the P40s or any v1 job waits for RAM.
5. **ETA (EST): sealed 10-16 about 17:00-21:00Z** (range 10-16 12:00Z to 10-17 06:00Z). That is probably after D2's first step. **D2 does not wait.** d2_plan uses the clean registry only if it is fully sealed when the D2 data config is written; otherwise D2 runs on the D2-X1 span mask as decided. If D2 starts first, the clean version serves D4 replay and stage T.

---

## 1. The rule

`d2_clean_repack.py` 1.0.1 implements this rule. Every set is read from the pinned decisions file `$Q/D2_DECISIONS_2026-10-15.json` (`c8d47f22…`), not retyped.

- **X:** the documents that must not appear.
  - The canonical held-out split: six Codex packs, 5,391 ids. Joined to registry rids by the `d2_holdout_eos_audit` rule, that gives 5,055 rids.
  - Plus the BFCL-62 source rids (all 62 are already held-out).
  - Plus the 7 A2 3(a) text twins (6 of them are not held-out).
  - Total: **5,061 documents, 3,180,411 tokens.**
- **A (affected rows):** every sealed row holding a token of an X document: **3,386 rows**. This includes all 76 A2 3(a) rows.
- **K (kept rows):** the other **1,497 rows**, in 886 runs. They are used as they are.
- **Re-pack:** every non-X document with any token in an A row, **whole**.
  - That is 22,970 documents, 12,349,164 tokens.
  - Its tokens are read from the sealed shards, never re-tokenized.
  - Documents keep the sealed allocation order (`documents.jsonl` `doc`), which keeps the route interleave.
  - They are packed with `distill_corpus_build.stream_pack` itself: concatenated and cut every 4,096 positions, so a document may continue into the next row.
  - This gives 3,015 rows. Only the last row is padded, by 276 positions (separator id, doc_index -1, route 255).
  - The files use the sealed tensor names and metadata keys. Each document keeps its provenance: `source_doc`, `source_start`, source rows and `source_class`.
- **Boundary documents:** 1,737 re-packed documents also have tokens in a K row (1,660,519 tokens).
  - They are trained once, whole, in the re-pack.
  - In their K rows they stay as plain train-text context, and every term is masked there (`clean_all_mask.json`).
  - The kept-row all-terms mask is exactly the 1,660,520 kept-row positions owned by an X or re-packed document, under the `d2_mask_build` POSITION CONVENTION.
- **Masks carried by document:** the A2 and D2 rules are about documents, so they move with the document.
  - **A2 3(b):** 692 documents. They are read off the pinned v1 `all_mask.json`, whose spans must be an exact union of whole documents (refused otherwise).
  - **A2 3(c):** 8,112 documents, read off the pinned v1 `gt_mask.json` the same way.
  - **D2-X2 EOS:** 5,498 flagged train documents. That is 6,936 flagged minus 1,438 held-out. Each gets CE/EOS/UL off on [e-4, e-1), and the tool refuses any document whose tail is not `248046, 198, 248044`.
- **Row exclusion:** `clean_exclude_rows.json` lists the A rows per sealed file. It also carries v1's non-registry entry (K3 pass 2 row 10) unchanged, so it replaces v1's file for every D2 source.
- **Token conservation is asserted and re-verified:** X 3,180,411 + re-pack 12,349,164 + whole-in-K 4,471,193 = 20,000,768, which is every sealed position.

**Why boundary documents are re-packed whole.** The alternative was re-packing only their affected-row fragments.
- A head fragment would end without a separator and run into the next document.
- Adding a separator would teach a false EOS.
- Whole documents cost 2.57M tokens of re-pack instead of 0.91M, about 15% more teacher time. In return every train document is complete, and the packer convention stays the sealed one: a document is cut only at a row end.

## 2. The manifest (`$OC/distill/corpus/text_clean_v1/`, 120 MB on root)

| File | sha256 |
|---|---|
| `manifest.json` (file) | `1455f6284617f787351b98a993f36ff7451a54d342ba3b15453c1aaf05176632` (manifest_sha256 `5f25c817…`) |
| `documents.jsonl` | `848cc07f2e1f4ca38c1008f051b8a067ac536f37fe345a8d300d190c9880779b` |
| `clean_exclude_rows.json` | `40304d9d804ad312203a93af822026a748b1c168eb16a2220ebe4d17e545d996` |
| `clean_all_mask.json` | `7bc14ae22c79127a1d5b9f3a3829189288b98fef413f9b1669c6bc070a951b73` |
| `clean_gt_mask.json` | `e3406d154ffcf1839fcd0359d04e45f4fb819fdfefae627092e09757fd35b3a3` |
| `verify_receipt.json` | `14dedacef725bb0332bc06bac1708dc7b882e1a83c67b4c078ff9b1ca58a4c34` (ok, problems []) |
| `text-clean-00000` … `00010` | 256 rows each, 1,048,576 tokens each: `01d99603…`, `1b5ecfc1…`, `36c04a21…`, `ee3d6958…`, `80644673…`, `be62d718…`, `c52ae817…`, `f6d77e91…`, `79abb869…`, `d960ab6a…`, `18bee8fb…` |
| `text-clean-00011` | 199 rows, 814,828 tokens: `9fbbb536…` |

The full shas are in `manifest.json` `shards`.

**Counts printed by the build:**
- KD positions: 4,408,053 in kept rows plus 12,167,380 in the re-pack, **16,575,433** in total.
- All-terms mask: 1,723,659 kept-row positions and 181,784 re-pack positions.
- GT mask: 877,616 kept-row positions and 2,402,471 re-pack positions.

## 3. Teacher cache: card choice and evidence (receipts `$OC/distill/parity/d2clean-p40-2026-10-15/`, copied out of scratch, with `SHA256SUMS`; `vs_fp32_ref.json` `2ef133bd…`)

| Probe on GPU3, 8 sealed rows (text-00000 rows 0-7), frozen 1.0.5 | Result |
|---|---|
| `--linear-attn-kernel auto` (fla), Pascal venv | Refused by the card: `ptxas … Feature '.bf16' requires .target sm_80 or higher` |
| torch kernel, bf16, mb8 / mb4 | Out of memory (math SDPA scores of 12 / 6 GiB at T = 4,096) |
| **torch kernel, bf16, mb2** | **53.21 tok/s**, peak 20.9 GB, finite, sealed. Host MemAvailable fell about 10 GiB (the RSS is mostly the mmapped teacher) |
| torch kernel, fp32, mb1 (reference) | 47.26 tok/s, finite, sealed |

**Distance to the fp32 reference, on the same 32,768 positions** (KL(ref ‖ cache) over the reference's top-64):

| Cache | KL mean | KL p99 | Positions > 1 nat | Top-1 agreement | lse diff p99 | KL max |
|---|---|---|---|---|---|---|
| P40, bf16, torch GDN | 0.0116 | 0.049 | 59 | 0.9891 | 0.86 | 21.8 |
| Sealed 3090, bf16, fla | 0.0109 | 0.051 | 66 | 0.9879 | 0.89 | 38.0 |

- The worst positions are the same for both caches (row 2 pos 3,400-4,017 and row 6 pos 2,620-2,667). They are bf16-fragile positions of the teacher, not a kernel defect.
- P40 against the sealed cache directly: top-1 0.987, KL mean 0.012.
- Per owner policy 2026-10-06 all of these are diagnostics. The functional checks were finite output and identical ids/positions, and both passed.
- The kernel is recorded in every cache's `cache_config` (`"chunk_gated_delta_rule": "torch"`), so D2 receipts can name it.
- I did not use fp32: it is slower on the P40, and the sealed caches are bf16.

## 4. The job, and how it yields to v1

`$T/d2clean_recache.py` 1.0.1 runs one runner per card. Standard library only.

**What it runs:** the frozen `qtext_teacher_cache-1.0.5.py` (`f2465a11…`) with engine `tps_probe.py` (`449eecc4…`).

**Flags:** `--dtype bfloat16 --attn-implementation sdpa --max-batch-tokens 32768 --batch-order length --rows-per-shard 64 --head-precision fp32 --digest-threads 8 --teacher q38-27bu --host-resident off`, plus the P40 settings `--linear-attn-kernel torch --micro-batch 2`.

**Output:** `--out-root /mnt/lexar/omni-campaign/teacher_cache/qtext`. The cache of file F lands in `q38-27bu/<sha256(F)>`, where d2_plan looks.

**Gates**, re-read before every file and every 30 s while a file builds:
- **GO file:** `$Q/d2clean.go` (`53760d41…`): `cards=3`, `cards_after=2026-10-15T21:40:00Z:1,2`, with the manifest sha.
- **Clean state:** built and verified.
- **Card:** under 2,000 MiB used, no compute app, no live foreign `p40.claims` line (read through `d1_seal_lib.claim_state`), and `$Q/gpuN.lock` held only while a file builds.
- **v1 pressure:** read through the fold watchdog's own read-only `x3_v1_yield_watchdog.py --probe`. The runner yields on:
  - every v1 RAM wait (gpu6 queue, D1, post_d1, d1_seal, d2);
  - a K3 job queued below 75 GiB;
  - on a P40, the bank v1 review waiting for or holding the P40s.

  `swap_full` alone is recorded but not obeyed. It is the steady state while D1 and K3 run, and this job holds little anonymous RAM.
- **RAM:** MemAvailable must be at least 60 GiB to start, and the runner yields if it falls below 24 GiB.

**How it yields:**
1. SIGTERM the build. The tool turns it into exit 143, writes its "after" digest and keeps every finished 64-row shard. A yield loses at most one shard, about 1.4 h of one P40.
2. Release the claim line and the locks.
3. Wait for 600 s with no pressure before taking the card again.

It never sends SIGKILL. A build that fails twice on one file halts its runner with HALT in the status file.

**Live state (03:43Z arm, receipts `$OC/distill/parity/d2clean-p40-2026-10-15/arm-20261015T034316Z/`):**
- Runners: pids 2029329 (GPU3), 2029330 (GPU1) and 2029331 (GPU2).
- GPU3's gates were open at arm time, so it starts after the 600 s hold.
- GPU1/2 print `WAIT go: card N not in cards=[3]` until 21:40Z. After that they wait for the tool-call claim and the card to clear.
- Status: `$Q/d2clean.status`. Logs: `$Q/logs/d2clean/`.
- To stop: remove the card from `cards=` in the GO file, or `kill -TERM <runner pid>`. Either way the runner yields cleanly.

**ETA (EST):**
- Work: 12.35M tokens / 53.21 tok/s = 64.5 P40-hours, plus about 5% for digests and loads.
- Before the review: GPU3 runs from about 03:53Z on 10-15, and GPU1/2 join at about 21:50Z on 10-15. The D1 review takes the P40s at about 10-16 05:30Z (range 01:30Z-10:30Z, from the wave-4 D1 pace plus about 3 h of seal). That is about 41 P40-hours, roughly 7.1-7.8M tokens.
- After the review: the remaining 4.5-5.2M tokens take about 8-9 h on three P40s, once the review releases them (its length is unknown).
- **Sealed about 10-16 17:00-21:00Z** (range 10-16 12:00Z to 10-17 06:00Z).
- Any v1 RAM wait (for example the K3 B2 fp32 pass needing 93 GiB) pauses the runners and moves this later.

## 5. How D2 uses it (contract for L-TRAIN / d2_plan; I did not edit d2_plan)

1. **When the D2 data config is written:** run `d2_clean_repack.py state --out $OC/distill/corpus/text_clean_v1`.
   - **Exit 0** means built, verified, and 12/12 caches sealed (finite, structural, teacher steady). Take the clean mode (step 2).
   - **Exit 10, or anything else:** take the span mode, D2-X1 exactly as decided. **D2 never waits for this lane.**
2. **Clean mode:**
   - The 20 sealed registry sources take `exclude_rows=clean_exclude_rows.json`, `all_mask=clean_all_mask.json` and `gt_mask=clean_gt_mask.json`.
   - The 12 `text-clean-*` files are added as registry sources (kind text, `qtext_cache=<root>/q38-27bu/<file sha>`), with the same three files. Every entry is keyed by corpus sha, and the trainer's `load_row_exclusion` / `load_position_mask` read exactly this schema.
   - The K3 sources can take `clean_exclude_rows.json` too: it carries v1's K3 row unchanged.
   - Budget: the clean registry has 16,575,433 KD positions, below the 18.85M draw. D2 draws all of it, once, and the refill target is computed on that basis. The span-mode figure of about 2.96M uses a different accounting, so the lane prints both.
   - `D2_DECISIONS_APPLIED.json` names the mode, the manifest sha, the three mask shas and the 12 cache manifest shas. The Codex val/test label becomes "excluded from D2 inputs; not context-exposed".
3. **Span mode:** unchanged. The clean version is still the preferred registry for D4 replay and stage T. That replaces the "open option for stage T" line of `D2_DECISIONS_2026-10-15.json` `not_adopted`.
4. **Suggested A2 text** (the d2prep lane owns A2, and I did not edit it): "3(d') alternative to the D2-X1 span mask, used only if sealed before D2's data config: the registry rows holding any held-out / BFCL-62 / twin token are excluded and their other documents re-packed whole (`text_clean_v1`, manifest `5f25c817…`), with kept-row positions of re-packed documents and A2 3(b)/(c) and D2-X2 carried by document; teacher Q38 top-64 by the frozen qtext 1.0.5, P40 torch-GDN kernel recorded per cache."

## 6. What I could not verify

- **The sustained P40 rate on 256-row files.** 53.21 tok/s is from one 8-row shard. Three runners share one page cache of the 55.6 GB teacher. If a K3 pass takes about 93 GiB, the page cache may not hold the teacher, and the runners would read the root SSD and slow down. That would not slow v1, because the runners yield on its RAM waits.
- **The real interplay with the D1 bank v1 review.** The yield path is tested on a fake build and the seal queue's status format, not on a live review.
- **The first sealed file.** The first START was checked: 03:53:34Z, `text-clean-00000`, qtext pid 2198505 at 20.9 GB on GPU3, teacher tree `7e0fb8f0…` equal to the sealed caches' tree, `gpu3.lock` held by runner 2029329. No file had sealed when this was written.
- **Every ETA** in section 4.
- **The fp32-reference comparison covers 8 rows** of one file.
- **d2_plan has no clean mode yet.** It needs the change in section 5 before the clean version can be used.

## 7. Tools and tests

| File | Version | sha256 |
|---|---|---|
| `tools/d2_clean_repack.py` | 1.0.1 | `1d03d3e852e6d9d0dbcf501b79c0294124703692c87adb3ec8bca288bda26664` |
| `tools/test_d2_clean_repack.py` | | `79d97756a6870145f4d8e027aa15add5331482c40b46e4a4581dad6e73141394` |
| `tools/d2clean_recache.py` | 1.0.1 | `9891f32a2583606ea6610f154625af0342e6eebd5498a3ec425e85368c4c9c37` |
| `tools/test_d2clean_recache.py` | | `bb5e7ccbbd0de44da5ff8ea8f1782aa92b7fd2d89365d4f6c5f6c3d560f95adf` |
| `tools/_superseded/d2_clean_repack_1.0.0.py` | 1.0.0 | `5a50226e…` (refused its first real build on the decisions file's `"<path> (<note>)"` source string) |
| `tools/_superseded/d2clean_recache_1.0.0.py` | 1.0.0 | `5c9fbdff…` (P40 micro-batch default 4, which runs out of memory; never run) |
| `queues/frozen/qtext_teacher_cache-1.0.5.py` | 1.0.5 | `f2465a11…` (copy of `tools/_superseded/qtext_teacher_cache_1.0.5.py`) |

Local and server copies are byte-identical. `TOOL_PINS.json` was regenerated after the deploy (`TOOL_PINS VERIFY: 179 ok, 0 drift, 0 missing, 0 unpinned`). Other lanes regenerate it too; the local mirror is the server copy as of about 04:05Z.

**Test results:**
- `16 passed` (8 re-pack + 8 runner), run from `$T` and its conftest. Re-run on the deployed 1.0.1 at arm time: `16 passed in 5.39s`.
- Self-test: `SELF-TEST PASS (10/10)`.

**Negative controls:**
- The new re-pack tests against 1.0.0 give `3 failed, 5 passed`.
- A runner whose claim line says "released" gives `1 failed, 7 passed`.
- A runner that obeys no v1 signal gives `2 failed, 6 passed`.
- In-suite, a planner that forgets the boundary documents' kept-row positions is caught by `verify` ("misses").
