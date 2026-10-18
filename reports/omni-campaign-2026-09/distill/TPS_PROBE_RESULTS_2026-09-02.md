# TPS PROBE RESULTS — 2026-09-02

The one measurement that gates every stage-D budget. Run book: `CAMPAIGN.md` sections 1.8, 2.1 slot G0-12, 2.3 slot G4-2, 4.2 (inputs item 2), 5.2 row `tps_probe.py`. Constraints: `phase0-reader-digests/4_R5_training.md` section 4.

Tool: `distill/tps_probe.py` (sha256 in every receipt). Receipts: `distill/tps/<cell_id>.json`, one per cell, byte-identical under `OC/distill/` and `/home/cereal/omni-campaign/distill/`. Every number in the tables below is read out of a receipt by `distill/tps_summary.py`; the prose is the only hand-written part.

**Nothing here is a gate.** This file reports rates and the memory they cost. The one threshold it touches — D-R5-1, "27B LoRA below 100 tok/s collapses the trunk update to graft + last-N-layer LoRA" — is Cyril's to rule on. The probe supplies the number, not the verdict.

TABLE_HEADLINE_PLACEHOLDER

---

## 1. What a cell is

One (model, mode, dtype, sequence length, micro-batch, accumulation) point, measured twice on one pinned card:

* **forward-only** — trunk forward under `no_grad`, then the LM head over every position plus a top-64 gather. This is the teacher-logit rate, so the head and the gather are inside it; a "forward" that stops at the last hidden state would flatter the teacher-cache budget by the cost of a 248,320-row head.
* **forward + backward + optimizer step** — the student rate, including the optimizer step, the gradient clip, and (where it applies) the loss-scale check.

Three warmup steps are discarded and at least ten are timed. Warmup is not decoration here: the first optimizer step allocates the whole Adam state (7 GB at 0.8B, 54 GB on the host at 4B), Triton autotunes the `fla` chunked kernel on its first call, and the caching allocator grows once. In pass A the first 0.8B train step took roughly sixty seconds and the timed steps took 0.6 s.

## 2. What was measured

TABLE_MAIN_PLACEHOLDER

## 3. Derived budgets

GPU-hours are wall-clock times the number of devices the run occupies — the same definition `stage2_training_receipt.py` seals, so a stage-D training receipt and this probe can be compared without a unit argument. A two-card cell therefore burns GPU-hours at twice its wall-clock rate.

TABLE_BUDGETS_PLACEHOLDER

## 4. What the numbers say about the campaign's open decisions

NARRATIVE_PLACEHOLDER

## 5. Assumptions — all of them, stated

1. **The objective is cross-entropy, not the stage-D KD objective.** Loss is token-level CE on the input ids. The real stage-D term is a top-k forward-KL plus a tail bucket over the *same* logits, so it costs the same order and somewhat more (an extra gather and a KL reduction per chunk, plus reading the teacher cache from local disk). Every train rate here is therefore an **upper bound** on the rate of the real objective at the same shape. On-policy GKD steps are not modelled at all: they add a teacher forward on the student's own continuation and are priced separately from the 27B forward-only rows.
2. **The corpus is uniform random token ids** in `[0, 248044)` — the tokenizer-defined rows. Throughput at a fixed tensor shape does not depend on token values, but a real corpus brings packing, padding and variable-length segments, none of which are modelled. A packed 4,096-token batch with `cu_seqlens` will differ; how much is unmeasured.
3. **The head is run in chunks over the sequence** (`loss_chunks`, recorded per receipt), building one chunk of logits, backwarding it into a detached copy of the hidden state, freeing it, and backwarding the trunk once at the end. `--self-test` checks the value and the gradient against a one-shot `F.cross_entropy`. This is arithmetically identical to the one-shot path and materially cheaper in memory, so **the memory figures here are lower than the R5 digest's one-shot arithmetic predicted** (a 4,096 x 248,320 fp32 logit tensor is 4.07 GB before its gradient; one chunk of 512 positions is 0.5 GB).
4. **Gradient checkpointing is on in every training cell.** Rates would be roughly 20–35% higher without it at 0.8B, and nothing above 0.8B would fit at all. It is recorded per receipt.
5. **The vision tower is resident and frozen** in every cell except the 27B split, where it is dropped (`vision_tower_dropped: true` in that receipt) to buy about 1.2 GiB on the card that was short. A stage-D omni student **would** carry the tower, so the 27B split row is optimistic by roughly that much memory — not by any measurable throughput, since a text-only batch never calls it.
6. **"Full parameter" means the text trunk plus the LM head**, with the vision tower frozen. Parameter counts are in the receipts. The live parameter count differs from the safetensors-header count the ledger uses (0.8B: 852,985,920 live against 873,438,784 in headers; 4B: 4,539,265,536 live, which matches the calibration lane's figure exactly). This probe reports live counts; the ledger's header counts remain the ledger's.
7. **`sdpa` is the attention implementation everywhere** (recorded per receipt). `flash_attn` is absent from the venv; `eager` was not measured.
8. **The kernel path is imported, not re-implemented.** `kernel_path_report` comes from `calibration/oc_calib.py` and that file's sha256 is sealed into every receipt, so the field cannot drift away from the calibration lane's definition. Every cell measured `chunk=fla-0.5.2; recurrent=fla-0.5.2; conv-prefill=torch; conv-decode=torch` — the same v2 string the calibration lane reports for the 4B parent. The Gated-DeltaNet chunked kernel is Triton; the causal convolution is the torch fallback, because `causal_conv1d` is not installed.
9. **The 27B streamed teacher rate is measured with a warm page cache.** The host has 251 GiB of RAM and `Cached` went from 123.7 GiB to 181.4 GiB across the run, so the 51.75 GiB checkpoint sits in page cache after the first pass. A cold first pass is disk-bound and slower; the receipts carry `page_cache_gb_before`/`after` and the per-step spread so a reader can see it. In steady stage-D use the cache stays warm, so the warm number is the operationally right one — but it is not a cold-start number.
10. **The 27B split runs fp16 end to end**, not "bf16 with fp16 on the Turing half". A `device_map` split carries one dtype through the residual stream; mixing would insert a cast at every boundary. sm75 has no bf16 tensor cores, so fp16 is the only fast choice, and the dynamic loss scaler in the receipt is the price. LoRA adapters stay fp32 and cast themselves into the compute dtype in the forward, which is what makes them their own fp32 master copy.
11. **Budgets are linear extrapolations** of a ten-step median at one shape. They assume no data-loading stall, no checkpoint-write stall, no eval pauses, no restarts, no contention from another lane, and a corpus already on local disk. The campaign's own launch discipline (rolling 30-minute safety checkpoints, rung seals, free-run probes every 1M tokens) is *not* in these numbers; a real run will be slower than the rung table by whatever that overhead is.
12. **One GPU-hour of a 3090 and one GPU-hour of an RTX 8000 are counted the same.** They are not the same thing. The two-card rows spend half their GPU-hours on a card with no bf16 tensor cores and no flash-attention backend.

## 6. What did not work, and what it cost

FAILURES_PLACEHOLDER

## 7. Files

FILES_PLACEHOLDER
