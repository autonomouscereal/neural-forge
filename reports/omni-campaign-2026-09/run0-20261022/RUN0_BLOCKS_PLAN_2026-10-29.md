# Run 0 on the block layout of record (2026-10-29)

**Ruling:** `cyril-run0-block-layout-2026-10-29`. It sits on top of `cyril-run0-plant-qwen-2026-10-22`.

**Code:** chimera-omni `claude/run0-blocks-20261029`. The data loop that feeds it is chimera-omni `claude/run0-gen-20261028` (section 8).

**Status at 2026-10-29 ~22:00Z:**
- Built, with CPU tests green.
- Deployed to `/home/cereal/omni-campaign/chimera-omni-v2`.
- The plant is running.
- The P40 smoke and the readiness gate are next.
- Nothing has launched.

## 1. What Cyril asked for (verbatim intent)

> [Cyril quote withheld]

> [Cyril quote withheld]

> [Cyril quote withheld]

> [Cyril quote withheld]

> [Cyril quote withheld]

> [Cyril quote withheld]

> [Cyril quote withheld]

## 2. The layout (`co3-2b-run0`)

**Overall:**
- 7 blocks of `[gated_gqa | gdn_kv | kda_kv | sparse_gqa]`, so 28 layers. That is deeper than Qwen3.5-2B's 24.
- Width 2048. SiLU-GLU FFN 5120.
- Tied vocabulary 248,320 + 512.
- **2.071B parameters.** The old co2-2b was 1.923B.

| Layer | Module | Heads | Keys / values | Notes |
|---|---|---|---|---|
| 1 full attention | `GatedGQAAttention` | 32 q x 128, 2 KV | k != v | fp32 scores, QK-norm, partial mRoPE 64 dims, sigmoid output gate. Carries the bidirectional media refinement. A **gated residual attention** sits in front of it in blocks 2-7. |
| 2 Gated DeltaNet | `SharedKVDeltaAttention("gdn")` | 32 q over 2 states of 128x128 | k = v | One projection per state head. The key is its L2 form, the value the projection itself. Scalar decay per state head (Qwen's `-exp(A) softplus`). |
| 3 Kimi KDA | `SharedKVDeltaAttention("kda")` | as layer 2 | k = v | Per-channel decay `-5 * sigmoid(.)` (Kimi's form, rank-64 lift). |
| 4 sparse GQA | `SparseGQAAttention` | 32 q x 64, 2 KV | k = v | DeepSeek-V4 style: one cached tensor per KV head. K is its normed, rotated form; V is the raw projection. No latent compression. A QSA-shaped indexer covers the rest (below). |

**The layer-4 indexer:**
- 4 index heads x 32 dims over 4-token blocks.
- Each query keeps the top 2,048 tokens, plus the first block and a 128-token local window.
- Below 2,048 tokens the layer is exactly dense.
- The indexer trains by a KL to the layer's own attention mass at 64 sampled queries (DeepSeek DSA's warm-up rule, weight 0.1). This is injected as a gradient, so it works under activation checkpointing. The indexer input is detached.

**Gated full residual attention:**
- A pseudo-query, starting at an even mix, attends over the embedding and every earlier block boundary.
- A rank-128 per-channel sigmoid gate, with bias -4 (0.018), writes the mix into the residual. The model therefore starts as a plain residual.

**Kernels:**
- Pure torch, so they run on SM61.
- The chunked k = v delta rule uses the UT/WY form:
  - scalar decay is exact, at chunk 64;
  - per-channel decay is factored around the chunk midpoint at chunk 32, so every factor stays below e^80 at the -5 floor.
- Tests pin both kernels to the per-position recurrence, in outputs, final state and gradients, including the decay floor in fp32.

## 3. Interpretations, and the trade-offs I chose (lead calls; Cyril can revise)

1. **"Qwen sparse GQA" = Strata's QSA shape**: an indexer over 4:1-pooled keys picks the top 2,048 tokens, then dense GQA runs over them. "No compression" is read as: the stored k = v stays raw, while the indexer keeps its pooled view for choosing.
2. **k = v in the delta layers.** If v = k literally, the state can only store "which past keys look like this one". So the key is the L2-normalised projection and the value is the same projection unnormalised. It is still one projection and one cache. Whether this costs quality against k != v is unmeasured; the run-0 dumps will show it.
3. **2 KV heads in a linear layer = 2 states of 128x128.** That is 8x less state than Qwen's 16 heads. 32 query heads read them. I kept Cyril's 2 rather than raising the head dim, because a 256 head dim would have cost about 100M parameters per block.
4. **Size.** The 4-P40 plan keeps an fp32 replica of weights and grads on every card. 1.92B fit at a 20.7 GB peak.
   - The new layout is 2.07B, about +1.2 GB per card.
   - FFN 5120 was chosen over 4608 so less of Qwen's FFN is pruned.
   - If the P40 smoke peaks above 22.5 GB, the fallback is FFN 4608 (1.98B), which is one config field.
5. **Depth before width.** 28 layers at width 2048, not 24. Width stays 2048 so Qwen's embedding and FFNs can be planted.

## 4. The plant (`scripts/plant_qwen35_2b_blocks.py`)

**From Qwen/Qwen3.5-2B** (revision 15852e8c, downloaded 2026-10-29, Apache-2.0):
- the embedding (tied head), with run 0's control-row starts;
- the final norm and every block's two norms;
- every FFN. New layer `l` takes Qwen layer `round(l * 23 / 27)`: an even spread, four of them used twice, as in depth up-scaling. It is pruned 6144 -> 5120 by weight-only importance `|gate_i| |up_i| |down_:,i|`. **The pruning is the ablation.**

**Fresh:**
- every mixer of the new layout;
- the residual gates.

The media head's last layer starts at zero.

## 5. Training (unchanged run-0 loop, area C)

**Stage 1** (`--freeze-trunk-stage1`, at most 400 steps / 18 h):
- Trainable: the fresh mixers, the residual gates, the bridge, the media head and the control rows.
- The planted FFNs, norms and text rows stay frozen.
- On this layout the mixers are new modules (`stage1_patterns`), at `--new-module-lr`.

**Stage 2:** everything trains, WSD schedule, the same data, KD and media losses as the rev-2 plan.

**New log fields:** `index_kl` and `sparse_kept`.

## 6. Readiness gate (Cyril 2026-10-27: "[Cyril quote withheld]")

S3 smoke on the 4 P40s (`run0_queue.sh --smoke`), then:
- adversarial code review;
- a soak of at least 90 minutes with a kill -9 and resume;
- a 10-day projection from the measured positions per second;
- READY: YES/NO;
- a 15-minute watchdog after launch.

## 7. Tests (chimera-omni `tests/test_run0_blocks_torch_optional.py` + trainer)

There are 20 new tests:
- kernels against the recurrence, both decay kinds, values and gradients;
- the KDA floor in fp32;
- mixer state carry-over across calls;
- sparse GQA: dense below budget; selection keeps sink, own block and window; the indexer learns only from its KL;
- the gate starts nearly closed;
- the 2B parameter count;
- manifest round-trip, with legacy manifests unchanged;
- the proxy trains, and cached decode equals full forward;
- the plant;
- the real run-0 trainer through both stages on `co3-proxy`.

The existing arch and trainer suites stay green. Two registry pins were updated for the new planted profile.

## 8. The data that feeds it (live since 2026-10-29 ~21:00Z)

**run0-strata-writer 1.3.0** (`strata-evolved`, 0.40 of the writer's Strata time):
- the self-taught known-winner loop over all of agent_memory (40k seeds), the corpus (5,362) and Cyril's prompts;
- modes: harder, deeper, what-if, experiment, finish, solve, expand, explain, question.

**run0-evolve-media 1.0.0** (`strata-evolved-media`):
- the same rule for image, video, 3D, speech and music, in and out;
- CLAP judges music; speech has an ASR read-back gate.

**Gate:** both wait on strata-gate's background admission. That is closed while GPU6 is at or above 70 C, and GPU6 idles at 67-70 C. **This is the throughput limit of the whole loop** (section 9).

## 9. Open for Cyril

- **GPU6's background thermal floor (70 C)** keeps Strata background work rare. Its idle temperature is 67-70 C. Raising the floor (e.g. to 75 C), or a background duty cap, would let the self-taught loop run. This is a Strata-lane setting, so it was not changed.
- **k = v in the delta layers** (3.2) and **2 states of 128** (3.3) are his calls as written. The dumps will show whether they cost anything.
