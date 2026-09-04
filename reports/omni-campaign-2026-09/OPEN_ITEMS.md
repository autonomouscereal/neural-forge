# OMNI CAMPAIGN — OPEN ITEMS

Decisions, claims and findings that outlive a single lane. Every row is tagged the
way `CAMPAIGN.md` tags numbers: **[verified]** = read out of a live service, a
receipt or a file this session; **[assumed]** = inference not yet receipted;
**[proposed]** = a decision line for Cyril, never a gate anyone invents.

Append new items; never rewrite an existing one in place — strike it and add the
correction underneath with its own date.

---

## OI-1 — GPU claims: all seven cards allocated to the Chimera Omni build

Opened 2026-09-02 by the datasets lane, closing run-book item **2.0 / 0.2**
("Confirm the GPU4/GPU6 claims through the relay ... Record the relay reply in
`OC/OPEN_ITEMS.md`"). Status: **claims held, peer informed, no peer approval on
the recovery requests.**

### What the relay says right now [verified — `cli.agent_comm --json claim list`, 2026-09-02]

| Resource | Owner | Status | Claim set | Fencing | Acquired | Lease expires |
|---|---|---|---|---|---|---|
| `docker:ai-server/gpu-slot-6` (RTX 8000) | claude | active, lease live | `79bb10d2-7a4f-4427-9186-0f092cb9cb66` | 3361 | 2026-09-01T21:03:32.907761Z | **2026-09-02T21:03:32Z** |
| `docker:ai-server/gpu-slot-5` (P40) | claude | active, lease live | `fa20491e-5af7-41c6-86bc-9c40ea99cf42` | 3362 | 2026-09-01T21:03:35.312024Z | **2026-09-02T21:03:35Z** |
| `docker:ai-server/gpu-slot-0` (3090) | codex | expired lease, `recovery_requested` by claude 2026-09-01T21:04:24.279913Z | `49f604d8-0db3-4a76-9f15-026dca11b70e` | 2564 | 2026-08-27T20:45:52Z | expired 2026-08-28T01:20:44Z |
| `docker:ai-server/gpu-slot-2` (P40) | codex | expired lease, `recovery_requested` by claude 2026-09-01T21:04:25.877599Z | `e2d0a28f-dd03-4f6d-a16e-bd45f069d9c3` | 2361 | 2026-08-27T10:22:39Z | expired 2026-08-27T11:31:38Z |
| `docker:ai-server/gpu-slot-3` (P40) | codex | expired lease, `recovery_requested` by claude 2026-09-01T21:04:26.804955Z | `c1d2680c-b7a3-46b9-bb4f-0177c45b23f9` (bundled with `comfyui_upscale_gpu3_sprite_v001` + `tcp:192.168.50.222:28198`) | 1631 | 2026-08-24T18:01:01Z | expired 2026-08-24T20:01:01Z |
| `docker:ai-server/gpu-slot-4` (3090) | codex | expired lease, `recovery_requested` by claude 2026-08-28T13:36:52.128618Z (task `8324ea96`, the earlier H3 benchmark) | `8915540a-cd93-41fb-b9b8-e97fb3b061d7` | 2567 | 2026-08-27T20:45:58Z | expired 2026-08-28T01:20:47Z |

Recovery-request reason on slots 0/2/3, verbatim from the claim rows: *"Codex
lease on docker:ai-server/gpu-slot-N expired days ago and the GPU is idle; Cyril
full-permission Chimera Omni build order 2026-09-01 (task
`6ff05586-bc7c-47c0-a0eb-8f15653d5f45`); standing rule: take all GPU power once
H3 workflows finish."* Every one of them still has
`recovery_decision: null` — **no owner or user decision has been recorded**
[verified].

### The physical allocation [verified]

Cyril's direct order, 2026-09-01 ~21:10Z, quoted in the relay message that
announced the takeover: *"[Cyril quote withheld]"* `docker stop comfyui_h3_gpu4` was run
against Codex's idle sprite worker (0% util for ~16 h holding 17.7 GB); the
container is **stopped, not removed**. All seven cards — GPU0/GPU4 (3090),
GPU6 (RTX 8000), GPU1/2/3/5 (P40) — are allocated to the Chimera Omni build
under relay task `6ff05586`. `nvidia-smi` at 2026-09-02T07:33 local shows every
card idle (GPU0 24 MiB, GPU4 15 MiB, GPU6 198 MiB, P40s 5 MiB each) [verified].

### What Codex was told, and what came back [verified — thread `d544f3fe-0d99-4ce9-8fd1-6e07326ebd65`, "h3-hybrid-media-upgrade", 50 messages]

* `97fa42f5` 2026-09-01T20:56:51Z (claude) — CLAIM + REQUIREMENT CHANGE quoting
  Cyril's full-permission directive and announcing the GPU4 claim.
* `fc5d77dc` 2026-09-01T21:07:39Z (claude) — STATUS: takeover executed, sprite
  container stopped not removed, "restart it yourself only after coordinating
  GPU use on this thread", all seven GPUs allocated, slots 5/6 granted,
  recovery requested on 0/2/3(/4).
* No Codex reply on this thread accepts, disputes or decides the GPU question.
  Codex's messages on 2026-08-28 (`c0a71175`) and 2026-09-02 (`15dfb69c`) both
  say it *cannot* approve recovery, release leases or acquire claims from
  unattended coordination mode — planning feedback only. Its one real decision,
  `45eee4cd` 2026-08-28T13:50:26Z, approved GPU5 recovery behind fencing
  barrier 2713 **and refused GPU4** on the grounds that "the user's current
  direction is to avoid the 3090s"; Cyril's 2026-09-01 order supersedes that.

### Open, and who has to move

1. **The two live leases expire 2026-09-02T21:03Z.** `claim renew` on
   `gpu-slot-5` and `gpu-slot-6` before then, or the campaign's own claims go
   stale exactly the way Codex's did. Owner: whoever runs the next GPU slot.
2. **Slots 0/2/3/4 are being used under an undecided recovery request.** The
   protocol says expired leases go through `recovery_requested`, approved by the
   owner or the user — never stolen. Cyril's verbal order is the user approval;
   it is quoted in the relay and here, but there is no `recovery_decision` row.
   [proposed] Ask Cyril to record one line of approval in the thread, or have an
   attended Codex run decide, so the registry state matches the physical state.
3. **`gpu-slot-1` (P40) is not covered by anything.** Its Codex claim set
   (`7a0b00a3-b63c-48c9-aa64-c249af59c3ed`, acquired 2026-08-27T10:32:23Z) is
   `status: active` with `lease_expired: true` and **no recovery request** —
   it was missed by the 0/2/3/4 sweep. GPU1 is the card carrying the export
   lane's in-flight 27B sweep (`oc-env-27b-gpu1-q4kv`, section 2.2 P-0), so the
   physical use and the claim ledger disagree there too. [proposed] file the
   fourth recovery request when the sweep writes its receipt.
4. **Codex's stopped container.** `comfyui_h3_gpu4` is stopped, not removed, and
   the campaign's standing rule is never to stop what you did not start. The
   stop was Cyril-ordered and is recorded here; restarting it is Codex's call
   and must be coordinated on `d544f3fe` first.

Until items 1-2 are settled, the campaign's own rule (`CAMPAIGN.md` section 0.5:
"GPU4 and GPU6 only after the relay claim is confirmed") is satisfied for
**GPU5 and GPU6 by a live claim**, and for **GPU0/2/3/4 by Cyril's order plus an
undecided recovery request** — which is a user authorisation, not a relay one.
Anyone reading this before running on 0/2/3/4 should know that is the standing.

---

## OI-2 — ~~The curated registry is 120 entries short of what it advertises~~ CORRECTED: the registry is complete

Opened 2026-09-02 by the datasets lane, out of run-book item **2.0 / 0.5** (the
bounded checksum walk). ~~Receipts: `OC/datasets/registry-walk-2026-09-02.json`
(sha256 `788def6abe6386eb21507e803efe4fc68330cb6c8743307b426c1fad89baa132`) and
`OC/datasets/registry-walk-plan-2026-09-02.json` (sha256
`f9a410e23d523fd6970c33defc040250975babf1254169cc4f3dcc1897985407`)~~ — those
receipts came from a walk that understood one of four registry schemas and are
kept only as the record of the error (pre-fix copies preserved under
`OC/datasets/.bak-20260902-141303/` and
`/home/cereal/omni-campaign/datasets/.bak-20260902-141629/`). **The live
receipts are the ones in the correction below.** Everything under this heading
tagged [verified] on the first pass was re-derived on the second; only the two
struck bullets changed.

* **Nothing is corrupt.** ~~505 of 505 readable~~ shard files reproduce the
  sha256 the registry publishes for them; 0 mismatches; the master pointer
  digests to
  `a71120e2c88ac620e96874c08cb93abf4ab51fb9e497d69b79104fde84ad731b`, which is
  exactly the value the local append receipt claimed and R1 had to mark
  [assumed] [verified]. (The denominator was wrong — see the correction below;
  the real figure is **640 of 640**.)
* ~~**120 of 631 entries have no shard index and no `shards.jsonl` on disk**
  (`file does not exist`), plus 6 pointers with no `shards_jsonl` field and 9
  individual shard files listed but missing. Reachable: 514 shards / 82,145 rows
  against the advertised 640 shards / 94,318 rows — **12,173 rows (12.9%) cannot
  be loaded through the registry today**~~ **[STRUCK 2026-09-02 — WRONG. This
  was a defect in `registry_walk.py`, not a fact about the share. See the
  correction below.]**
* **89 entries carry a stale `shard_index_sha256` in the master pointer**: the
  file on disk matches the per-dataset `SHARD_INDEX_POINTER.json`'s sha and not
  the master entry's copy (example: `kepler_broad_domain_knowledge3_ready_427`,
  file `316a8dbc…`, pointer says `316a8dbc…`, master entry says `92e6767b…`).
  The payload shards under those entries all verify — this is master-index
  metadata drift, not data loss [verified].
* ~~[proposed] Decide before stage T whether the 12,173 unreachable rows are
  (a) repaired by re-pointing the 120 entries at their real shard dirs,
  (b) dropped from the training arithmetic, or (c) declared lost. Every row
  count in the plan (`ready_untrained_candidate 52,416`, "94,318 rows") is an
  advertised number, not a loadable one, until this is ruled on.~~
  **[STRUCK 2026-09-02 — there is no such decision to make. Every one of those
  rows loads. See the correction below.]**

### Correction, 2026-09-02 (second walk, after the schema fix)

The finding above was a tool bug written into this ledger as a verified fact.
`registry_walk.py` understood exactly one of the registry's **four** live
shard-list schemas, and reported the other three as absent data. An adversarial
review refuted it by reading three named files on the share; the tool was fixed
and the whole walk re-run in 118 seconds.

Receipts for the corrected pass (byte-identical local and
`/home/cereal/omni-campaign/datasets/` [verified]):

| File | sha256 |
|---|---|
| `datasets/registry_walk.py` | `5d519fca8f393c5b42d7c209e7299d956f45fe195e84b8ae5887212f9c4db651` |
| `datasets/registry-walk-2026-09-02.json` | `81f8ba271f765e5fa4c26fb876e7aa6da314f74005658fb40d47adc404b95329` |
| `datasets/registry-walk-plan-2026-09-02.json` | `4d7842c777acc734d56427dfdaa263b10f90baac084582cf2adc40b3150a7ead` |
| `datasets/registry-walk-2026-09-02.progress.jsonl` | `e9b5c6cfc3548af3cae3ce86e0ed35d0c7b2d095e327d8bff20a943b9f43ffa1` |

**The curated registry is complete and intact** [verified]:

* **631 of 631 entries walked; 640 of 640 shards seen; 640 reproduce their
  published sha256; 0 mismatches, 0 unreadable, 0 unresolvable.** Rows read
  through the shard files: **94,318** — exactly the number the master pointer
  advertises. The plan pass independently lists 640 shards / 94,318 rows /
  539,954,956 declared payload bytes (0.503 GiB).
* **Nothing needs repairing, dropping or declaring lost.** The 12,173-row hole
  never existed: 11,993 census rows across 135 entries were recovered by the
  fix alone, and the remaining difference is the 2,334 rows that carry media
  sidecars with no extractable instruction/response text (counted, not lost).
* `master_shard_collection_sha256` as published by the pointer:
  `7c8cccd7dcace89189329bb9d9d99ff5ae1d921b62d830e31d7df6520746e646` (recorded;
  the collection-level digest itself is still unverified — see the gaps).

What the three schema generations actually were:

1. **120 entries name a `<dataset>.tmp/` staging directory** in their
   `SHARD_INDEX_POINTER.json` that was renamed to `<dataset>/` after staging.
   The real `shard_index.json` and `shards.jsonl` sit in the directory the
   pointer itself was read from and reproduce the pointer's own published
   digests; 5 of them carry stale `.tmp` payload paths as well. 10,985 rows sit
   behind them and every shard verifies. Recorded now as `pointer_path_stale`
   (240) and `shard_path_stale` (5) — a **metadata defect, repairable, no data
   loss** [verified].
2. **9 shard rows use `relative_path` / `bytes` / `rows`** instead of
   `path` / `byte_count` / `row_count`. The first walk read `path` as `""`,
   called `open("")`, and logged `file does not exist`. Example:
   `memory_batch_000001_agentic_memory_ready_6_append_current-000000.jsonl`,
   12,224 B, sha256 `2bf2de54c22ad697529df2b1186b424dbb0bfd3f1f39079e4accddb82195962b`
   — byte-for-byte the digest the first walk itself recorded while calling the
   file nonexistent [verified].
3. **6 pointers use a different generation entirely**: 5 are
   `omnicoder.shard_pointer.v1` (`train_jsonl` + `shard_index`, no
   `shards_jsonl`; the payload digest lives inside `shard_index.json`), and 1
   `SHARD_INDEX_POINTER.json` *is* an `omnicoder.training_shard_index.v1`
   document with an inline `shards` array. Both are now followed; the report
   records which route produced each shard list
   (`shard_list_sources: {shards_jsonl 625, shard_index_train_jsonl 5,
   pointer_inline_shards 1}`).

**The 89 stale `shard_index_sha256` finding stands unchanged** [verified]:
`(verdict_entry, verdict_pointer)` is `('MISMATCH','match')` for exactly 89
entries and `('match','match')` for the other 541, zero in the other direction;
every payload shard under those entries verifies. Master-index metadata drift,
not data loss. Example unchanged: `kepler_broad_domain_knowledge3_ready_427`,
file `316a8dbc…`, per-dataset pointer says `316a8dbc…`, master entry says
`92e6767b…`.

* [proposed] The only decision left for Cyril here is a **repair-or-leave** call
  on metadata: 120 pointers with stale `.tmp` paths and 89 master entries with
  stale index digests. Both are cosmetic for training — the loader can reach
  every row today — but a future tool that trusts the pointer verbatim will
  break on them exactly the way this one did.
* 28 shard rows declare no size at all (neither `byte_count` nor `bytes`);
  that is why `bytes_read` 545,926,020 exceeds `declared_payload_bytes_walked`
  539,954,956 by 5,971,064. Counted as `shard_row_without_byte_count` rather
  than left as an unexplained gap [verified].

---

## OI-3 — ~~No long-context row exists in the curated set, measured~~ CORRECTED: 24 rows exceed 16k tokens, none exceed 32k

~~Same receipt as OI-2. The Qwen3.5-4B tokenizer (`tokenizer.json` sha256
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`, vocab
248,070) over all 79,991 rows with extractable text: **35,544,653 tokens**, and
the longest single row in the entire registry is **15,787 tokens**
(`v9_base_validated_28025-000005.jsonl` line 2,889, route `math`). Rows at or
above 8,192 tokens: **3**. At or above 16,384: **0**. At or above 32,768:
**0**.~~ **[STRUCK 2026-09-02 — measured over 79,991 of 94,318 rows, because
12.9% of the registry was never tokenized by the buggy walk. The un-censused
population is exactly where the long rows were.]**

### Correction, 2026-09-02 (second walk, whole registry)

Same tokenizer (`tokenizer.json` sha256
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`, vocab
248,070, `add_special_tokens=false`), now over **all 94,318 rows** the registry
declares — 91,984 of them carrying extractable instruction/response text
[verified, receipt `81f8ba27…`]:

* **42,145,686 tokens** across **50** canonical routes (156,346,415 characters);
  **40,580,784** through the loader's 8,192-character compaction, with 688 rows
  truncated by it.
* Rows at or above **4,096** tokens: **571**. At or above **8,192**: **147**.
  At or above **16,384**: **24**. At or above **32,768**: **0**.
* The longest single row in the registry is **27,093 tokens** —
  `manual_cleanup_batch_000167_arc_agi_2_training_reasoning_ready_1000_append_current-000000.jsonl`
  line 33, route `text`.
* All 24 of the >=16k rows live in two entries the first walk never censused:
  entry 548 `manual_cleanup_batch_000167_arc_agi_2_…` (20 rows, max 27,093) and
  entry 581 `manual_cleanup_batch_000197_corrected_memoryaware_adaptiveclip_…`
  (4 rows, max 27,022). Both were behind stale `.tmp` pointer paths. The
  struck claim above was not merely imprecise; it was falsified by the very
  rows it omitted.
* 2,334 rows carry media/sidecar payloads with no extractable
  instruction/response text and are excluded from the token sums (counted, not
  diagnosed). **0** rows are refused by `training_allowed=false` /
  `use_policy=do_not_train` [verified].

The plan's ">=32k-token evidence corpus" **still has to be BUILT** — that part
survives, and is now measured across every row in the registry rather than 87%
of them. But the campaign should note that a 27k-token row does exist and that
the >=8k population is 147 rows, not 3: any stage-T ruling that assumed the
registry tops out near 12.6k characters is working from a number that was never
true.

Top routes by tokens: `text` 30,143 rows / 16,979,432 tok; `code` 11,169 /
5,803,199; `tool` 14,329 / 4,968,981; `long_context` 5,357 / 4,906,580; `math`
4,997 / 2,530,783; `reasoning` 6,211 / 2,430,534; `image` 4,228 / 1,170,960;
`music` 2,768 / 632,040; `ocr` 2,585 / 427,056; `web_knowledge` 1,110 / 403,234.

---

## OI-4 — A tool that understands one schema will call the other three missing

Opened 2026-09-02, out of the OI-2/OI-3 corrections. Worth carrying forward
because it is a process finding, not a dataset one.

* The first registry walk was well-built — bounded reads, one streaming pass,
  append-only fsynced resume, report as a pure function of the log — and its
  27-test self-test passed. **Every one of those tests emitted a single schema.**
  The suite could not have caught the bug because the fixture was written by the
  same assumption as the code.
* Worse, the fixture's damage case was a genuinely deleted file, which produces
  the identical `status:"missing"` / `"file does not exist"` signature as a row
  whose path could not be built. The assertion "shard_unreadable reported"
  passed for the wrong reason and actively concealed the defect.
* The suite now carries all four generations plus an unresolvable row and a row
  with no declared size, and asserts a **distinct problem kind** for "this
  reader could not build a path" (`shard_row_without_path`) versus "the file the
  registry named is absent" (`shard_unreadable`). Server run: **51 passed, 0
  failed** (was 27); workstation with `--no-tokens`: **40 passed, 0 failed**
  (was 20). Running the new fixture through the pre-fix module reproduces the
  production failure exactly: 4 of 8 shards matched, 3 "unreadable".
* [proposed] Standing rule for this campaign: when a walk reports missing data
  on a store nobody has audited, **read one of the missing files by name before
  the finding is written down as [verified]**. It cost two reads to refute this
  one, and it had already been escalated to a decision for Cyril.
