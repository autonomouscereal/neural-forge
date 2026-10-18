# K3 acceleration plan, 2026-10-14

**Written:** 2026-10-15 between 01:15Z and 02:45Z by Claude, lane k3plan. Inputs are the live server state at 01:14:51Z, the GPU6 queue logs, lane k3accel's 1.3.0 parity receipts, and the k3_teacher_cache 1.3.0 review. "EST" marks an estimate, "UNVERIFIED" marks something I did not see printed. All times are UTC. Host logs print UTC-6, and I converted them.

**What I changed on the server:** I added staged files only. Nothing was installed, launched, stopped or claimed. The GPU6 queue (pid 107948), its identity gate (pid 306480) and the lock bridge (pid 726165) were not touched. The same goes for GPU0/4 (D1), GPU1/2 (tool-call self-gen), GPU3 (fold research) and GPU5. The tests ran on the CPU and only read GPU temperatures and names.

---

## 0. Bottom line

1. **The biggest lever is GPU6's cooling, not code.** Under K3 load, GPU6 sits in SW thermal slowdown in 89-99% of polls, at a median SM clock of 300-345 MHz (its maximum is 1,620 MHz), and draws only 94-99 W of its 250 W limit. The card reports no fan, so it depends entirely on chassis airflow. One B2 layer ran while the card was still cool: 1,605 MHz and 2.4x the speed of the hot layers.
   - A cooling fix leaves the numbers bit-identical.
   - It takes 1.3.0 from about 19-34 h per pass to about 11-14 h (EST).
   - Section 7 lists options for Cyril. I changed no fan, BIOS, BMC or power setting.
2. **Install queue_gpu6 1.4.0 before the identity gate ends** (about 12:30Z on 10-15, EST).
   - It fixes the must-fix finding from the 1.3.0 review: after an out-of-memory error, 1.3.x resumes a pass at a different group size, which silently changes its MoE numbers.
   - It adopts the running identity gate and fp32 pass instead of starting them a second time.
   - Without a 1.3.0 review file it runs the B2-proven 1.2.4, exactly as now.
3. **Turn on 1.3.0 only after d2_plan accepts it** (install step 4). d2_plan 1.2.0 counts a K3 pass as sealed only when the frozen 1.2.4 sealed it. A pass sealed by 1.3.0, or on another card class, would leave D2 waiting forever at its first K3 row. Once that is fixed, 1.4.0 first runs the GPU6 bit-identity gate (about 1-1.5 h) and then the passes on 1.3.0.
4. **When D1 seals, choose K3-first or D2-first with numbers measured at that moment** (section 4.5). Without a cooling fix, K3-first on the 3090s finishes D2 about 1-3 days sooner: D2 done 10-18 05:00Z to 10-19 07:30Z, against 10-19 12:25Z to 10-22 01:35Z for D2-first. With GPU6 cooled, D2-first is as good or better. The 3090 pass rate is unmeasured; `k3_multicard.sh` with `eqcheck_only=1` measures it in about 2 h.
5. **The D2 curriculum change requested here already exists as amendment A2 item 8** (DRAFT, lane d2prep). It puts the K3 rows after the 20M rung, behind a hard seal gate. Section 6 gives two small additions for that lane, 8a and 8b, plus when D2 reaches its first K3 row: 10-16 23:05Z to 10-17 08:05Z on the upside engine, and 10-18 02:05Z to 11:05Z on the planning engine (both EST).

---

## 1. State at 01:14:51Z (read-only check)

| Item | Reading |
|---|---|
| GPU6 | B2 fold-identity gate, pid 306480, at L29 of 93 (01:11:20Z). 23,434 MiB, 82 C, throttle reason `0x20` (SW thermal slowdown), 110 W. The bf16 review pass sealed at 19:45Z. |
| Identity pace | L12 (22:07:54Z) to L29 (01:11:20Z): 17 layers in 184 min, about 10.8 min per layer. 63 layers are left, so it ends at about 10-15 12:30Z (EST). |
| B2 verdict | After identity come the fp32 pass (the bf16 pass took 3.4 h, 16:22Z to 19:45Z; fp32 on sm_75 is unmeasured, EST 3.4-5 h), the export and parity. Verdict about 10-15 16:30Z-18:00Z (EST). |
| D1 (GPU0+GPU4) | At L18 (step 220/230 at 01:15:22Z). Step-10 stamps run from L11 (21:13Z) to L18 (01:03Z): 7 layers in 3 h 50 min, 32.9 min per layer (17-43 min; gated-GQA layers are short). With 45 layers left, the last rung lands 10-15 22:15Z to 10-16 07:15Z. Adding the 3.1 h seal, D1 registers 10-16 01:20Z to 10:20Z (EST). Wave 3 assumed 23-25 min per layer; the measured pace is slower. |
| Host | MemAvailable 73.8 GiB of 251.7 GiB. **Swap is full** (SwapFree 200 kB). The identity gate's RSS is 48.7 GB. |
| Tool-call self-gen (GPU1/2) | 1,269,827 of 1,500,000 tokens, eta 1.54 h (01:14Z), so about 02:50Z. That frees about 19.5 GB of llama-server RSS. |
| GPU3 | Claimed by fold-research x3-08b (pid 772203) since 00:46:19Z. |
| P2 | Not run: every cell was refused (GPU0/4 held by D1, GPU6 has no slot). |
| Disk | `/` has 70 G free (floor 60); `/mnt/lexar` has 347 G free. |

---

## 2. Measurements

### 2.1 GPU6 thermals (receipts: `$LX/scratch/k3accel/plan/receipts/gpu6logs-20261015T0120Z.tgz`, `clockfit.json`)

| Run | Polls under load | Share in SW thermal slowdown | SM clock median / mean | Temperature mean | Power mean |
|---|---|---|---|---|---|
| B2 bf16 review pass (`b2.thermal.log`, 16:22Z-19:45Z) | 151 | 0.89 | 345 / 566 MHz | 81.0 C | 99 W |
| Fold-identity gate (`b2-identity.thermal.log`, from 19:51Z) | 223 | 0.99 | 300 / 415 MHz | 81.3 C | 94 W |

- `nvidia-smi -q -i 6` at 01:15Z:
  - SW Thermal Slowdown: Active.
  - Lifetime counters: SW thermal slowdown 93,908 s (26.1 h), SW power capping 138,987 s, HW thermal slowdown 81.9 s.
  - Thresholds: max operating 82 C, slowdown 84 C, shutdown 87 C.
  - Power limit 250 W (min 150, max 250). `clocks.max.sm` 1,620 MHz; application clock 1,230 MHz.
  - Fan: N/A, so the card has no fan of its own and depends on chassis airflow.
  - VBIOS 90.02.4E.00.03, PCI 88:00.0.
- Board: Supermicro X10DRG-OT+-CPU. `ipmitool` is not installed, and `/dev/ipmi0` is root-only, so I could not read chassis fan speed or the BMC fan mode.
- **Speed at cool clocks (one layer):**
  - B2 L01 KDA computed 45,056 positions in 42.2 s (1,068 pos/s) at 1,605 MHz and 254 W.
  - The mean of the hot KDA layers (n=66) is 103.1 s (461 pos/s) at about 475 MHz.
  - That is **2.4x**. L01 holds 1 AttnRes block, where late layers hold up to 8, so the true gain on late layers is somewhat lower.
- A per-layer regression of compute time against SM clock does not explain the variation (R² 0.09-0.53), because 60 s polls cannot resolve the clock inside a layer. I do not use it.

### 2.2 Rates per pass (positions per pass = 122 rows × 4,096 = 499,712; 69 KDA + 24 MLA layers)

| Source | KDA pos/s | MLA pos/s | Receipt |
|---|---|---|---|
| GPU6, 1.2.4, throttled: production `CachePass.layer` in the identity gate, 143,360 positions, 4 groups | 303-411 | 474-801 | `$Q/logs/gpu6/b2-identity_pass.log` (L01, L02, L12, L24; L03-L27) |
| P40, 1.3.0, 63,952 positions (16 rows × 3,997), 8 synthetic blocks | 946-1,048 | 1,760-1,816 at T ≤ 2,048 | `$LX/scratch/k3accel/runs/v130-20261014T200903Z/{w1_B,w1_C,w4_B,w4_C,w2_B,b2geo_B}.log` |
| 1.3.0 over 1.2.4, per layer at production group size (P40) | 1.15-1.46x | | k3accel report: 8-group layer 1.15-1.25x; steady L85 1.46x |

### 2.3 Card memory: does a pass fit 24 GB?

- **KDA layer, 1.3.0, on a 24 GB P40:**
  - Peak 23.09e9 B at 65,536 group tokens with `--moe-group-batch` 1 (w1).
  - Peak 23.98e9 B with four groups of 16,384 at m=4 (w4).
  - Both ran.
- **MLA layer:**
  - 24.05e9 B at T = 2,048 (w2).
  - **T = 4,096 runs out of memory on a P40.** sm_61 runs the math SDPA, which materializes the T² scores.
  - **So no P40 can run a production pass** without a new attention path, and the P40s are out of scope.
- **RTX 3090 (sm_86, 24 GB):**
  - The bf16 SDPA there dispatches to a fused kernel, so the score tensor is never materialized (EST, UNVERIFIED).
  - The KDA peak at 32,768 group tokens is below the P40's 65,536 peak (EST).
  - The first proof is the eqcheck run (section 5).

### 2.4 Host RAM per pass

- `k3_reserve`:
  - State is 499,712 × 7,168 × 2 B × (8 block slots + 2) = 71.6 GB, which is 66.7 GiB.
  - Adding 20 GiB of overhead gives **87 GiB**.
  - The RAM gate adds 6 GiB, so **93 GiB**.
- 1.3.0 adds a pinned layer arena of about 18-19 GB, and its end-of-layer RSS was 1.9-2.8 GiB above 1.2.4 (k3accel). That leaves the 20 GiB overhead tight for 1.3.0, so install step 3 raises it to 23.
- Box: 251.7 GiB.

| Combination | Host need | Fits? |
|---|---|---|
| 2 K3 passes + about 25 GiB baseline | 211 GiB | yes, after D1 |
| 3 K3 passes | 279 GiB | no |
| 1 K3 pass + D2 upside (about 94 GiB, design) | 212 GiB | yes, tight |
| 2 K3 passes + D2 upside | 280 GiB | no |
| 2 K3 passes + D2 planning (48.5 GiB) | 234 GiB | no (too close) |

---

## 3. Hours per pass, by option (arithmetic in `distill/tools/k3_schedule_projection.py`, output in `receipts/k3_schedule_projection.json`)

| Option | Per pass | Arithmetic |
|---|---|---|
| 1.2.4, GPU6 as now (throttled) | **29.8-43.9 h** | KDA 69 × 499,712 / (411…303) = 23.3-31.6 h; MLA 24 × 499,712 / (801…474) = 4.2-7.0 h; serial reads 93 × 75-190 s = 1.9-4.9 h; checkpoints 0.4 h. The gpu6 lane estimated 45-50 h. |
| 1.3.0, GPU6 throttled | **18.8-33.6 h** | (KDA + MLA) / 1.46…1.15; reads hidden by the loader; async checkpoints. Under a hard thermal limit the gain may be lower: the card gets busier and runs hotter. |
| 1.2.4, GPU6 cooled | 15.2-22.0 h | KDA at 1,068…750 pos/s (one cool layer; 750 is an assumed floor), MLA 853 pos/s (L03 at 1,035 MHz), plus reads and checkpoints. |
| **1.3.0, GPU6 cooled** | **11.3-14.0 h** | Assumes the cooled card is at least as fast as a P40 on 1.3.0 (its spec is higher): KDA 946-1,048 pos/s, MLA 935-1,816 pos/s, plus 0.3 h. |
| 1.3.0 on an RTX 3090 | 4.2-14.0 h (unmeasured) | 4.2 h is k3accel's spec-sheet scaling; 14.0 h is the P40 floor. |

---

## 4. Multi-card K3 schedule

### 4.1 The four passes are independent

Each pass is one corpus file, `pass-k`, with 122 rows. Its rows are independent sequences, and it writes `pass-k-of-4`. Two passes can therefore run on two cards at once with no coupling, but **each pass must run start to finish on one card class, one tool and one group-token count**. The bf16 MoE numbers depend on the GEMM row count (B2 diagnosis), so a pass must never mix geometries.

### 4.2 Which cards

| Card | K3? | Why |
|---|---|---|
| GPU6 RTX 8000 46 GB | yes (B2-proven) | Throttled; see section 7 |
| GPU0, GPU4 RTX 3090 | after D1 seals, and only if K3-first is chosen | They are D2's upside cards. They need the card-class check (section 5). |
| GPU1-3 P40 | no | MLA at T = 4,096 does not fit (section 2.3) |
| GPU5 | never | |

### 4.3 Host RAM limits K3 to 2 passes at once (section 2.4)

Two ways to lift that limit were considered and **not recommended now**:

- **Disk-backed AttnRes snapshots (mmap on `/mnt/lexar`).**
  - The AttnRes read touches every finished block slot at every layer: on average about 4 slots × 7.16 GB, plus the 17 GB layer, so about 46 GB per layer per pass through one USB SSD.
  - The SSD was measured at 90-440 MB/s, typically 200-300. That is 150-230 s of I/O per layer for a single pass. Three passes would share the drive: 3 × 46 GB / 250 MB/s = 552 s per layer, which makes them I/O-bound at about 14 h per pass.
  - It also needs a new storage path in the tool (a new sha, so new parity), and mmap writes on exFAT are untested.
  - Root cannot hold the 57 GB of slots (70 G free, 60 G floor).
- **Eight passes of 0.25M tokens.**
  - Footprint drops to about 54 GiB (35.8 GB of state plus 20 GiB), so 3 fit at once.
  - But weight reads double: 8 × 93 × 17 GB = 12.6 TB through the same USB drive, at least 11.7 h of reading in total.
  - The `pass-k-of-4` layout that d2_plan, xtok, the probe and the corpus manifest rely on changes.
  - The splits must fall on 16-row group boundaries with the same trimmed T to stay inside the proven geometry.
  - It only pays if 3 or more fast cards are free for K3 at once, which only happens if D2 waits anyway. It stays a fallback.

### 4.4 Timelines (EST; from `k3_schedule_projection.py`; B2 verdict 10-15 16:30Z-18:00Z; 1.3.0 gate 1-1.5 h)

| Option | K3 sealed | D2 done (upside engine) | Notes |
|---|---|---|---|
| O1: 1.2.4, GPU6 only, as now | 10-20 15:40Z to 10-23 01:50Z | 10-21 07:20Z to 10-23 17:30Z | The lane's 45-50 h per pass would push this to about 10-24 |
| O2: 1.3.0, GPU6 only, throttled | 10-18 20:45Z to 10-21 09:55Z | 10-19 12:25Z to 10-22 01:35Z | D2 waits at 20M |
| **O3: 1.3.0, GPU6 only, cooled** | 10-17 14:35Z to 10-18 03:25Z | **10-18 06:15Z to 10-18 19:10Z** | D2 runs on GPU0+GPU4 at the same time |
| **O4: 1.3.0, GPU6 pass 0 + 3090s K3-first** | 10-16 16:30Z to 10-17 19:05Z | **10-18 04:55Z to 10-19 07:30Z** | P2 first, then the 3090 check. GPU6 is capped at 1 pass. D2 starts after the seal. |

- In O4, GPU6's pass 0 ends at 10-16 12:20Z to 10-17 05:05Z. GPU0 runs passes 3 and 2, ending at 10-16 12:15Z to 10-17 17:50Z. GPU4 starts pass 1 once GPU6's pass frees the RAM and ends at 10-16 16:30Z to 10-17 19:05Z. At most 2 passes run at once.
- "D2 done" uses the upside engine: 20M tokens in 20.75 h, then 5.08 h for the rest of the short run (24.9M tokens), 4.2 h for the long rung, and a 6.4 h seal. All of these are design rates × 1.3.
- **Planning engine (split cell, GPU6 + GPU4):** D2 itself needs GPU6, so K3 cannot stay on GPU6 once D2 starts. Let GPU6 finish its current pass, set `$Q/k3_gpu6.max_passes` to the number of passes it has started, and run the rest on GPU0 with `k3_multicard.sh 0`. D2 then needs 47.75 h to reach 20M, and the RAM fits: 1 K3 pass + D2 planning = 141.5 GiB + baseline.

### 4.5 Decision rule at D1 seal (stated now, before the numbers exist)

1. Run P2 first (GPU0, about 1 h). Then run `k3_multicard.sh 0` with `eqcheck_only=1` in the GO file (about 1.5-2.5 h). It checks the 3090 class and prints the 3090's per-layer rate at the B2 geometry.
2. Estimate a 3090 pass:
   - Take the eqcheck's 93-layer compute seconds, multiply by 499,712 / 45,056 = 11.09, and add 10% for T = 4,096 attention (EST).
   - Take reads as max(0, 17 GB / 250 MB/s − compute per layer).
3. Estimate GPU6's remaining time from its own pass log (layers done per hour).
4. Compute both finishes:
   - **D2-first:** max(D2 start + 20.75 h, K3 sealed with GPU6 only) + 15.7 h.
   - **K3-first:** K3 sealed with the 3090s + 36.4 h.
5. Pick the earlier finish. If K3-first wins:
   - write `$Q/k3_multicard.go` with `gpus=0,4`, without `eqcheck_only`;
   - write `1` into `$Q/k3_gpu6.max_passes`, which hands the remaining passes to the 3090s;
   - GPU6 then moves on to H3 and j6 (wave-3 option 2 for free).

---

## 5. Card-class equivalence check: ruling `k3-card-class-equivalence-2026-10-15`

Recorded in `distill/STAGE_D_RULINGS_2026-10-12.md` (appended; backup `distill/backups/STAGE_D_RULINGS_2026-10-12.pre-k3-card-class-aaacf362.md` on both sides) and in `$Q/k3_card_class_ruling.txt` (sha `5da95bfb…`, which `k3_multicard.sh` reads). It was written before any result, under the standing GO and owner policy 2026-10-06. The ruling has four parts:

1. **GPU6 with 1.3.0 (same card class, another tool build): bit-identity is required.**
   - 1.3.0 seals passes on GPU6 only if `k3-v130-gate` shows that every prefix and block slot of w1 (L84-85), w2p (L83 at T = 4,096) and w4 (L84-85 in four groups) is bit-identical to the B2-proven 1.2.4 on GPU6.
   - Its +1-ulp negative control must report DIFFER.
   - Otherwise the passes keep 1.2.4.
2. **Other card classes: a functional bar plus diagnostics.**
   - The check is a full 93-layer bf16 review pass of the B2 review ids at the B2 geometry, run on that card class with the tool its passes will use.
   - **Functional:** the pass seals, and `parity` (1.2.4's code, `PARITY_THRESHOLDS` unchanged) against the B2 fp32 reference passes. This is the same bar GPU6's bf16 pass met.
   - **Diagnostic, never a refusal:** the same parity read against GPU6's own bf16 pass and against the fold dump.
   - Bit-equality across card classes is not expected and not required.
   - A FAIL refuses that class; it never loosens a threshold.
3. **One pass, one geometry:** one card class, one tool and one group-token count per pass (`<pass>.geometry`). Mixed classes are allowed in the cache and are recorded in `$Q/k3_pass_tools.json`.
4. **Out of scope:** the P40s and GPU5.

Cost: the check is read-bound, about 93 × 17 GB at about 250 MB/s, so 1.5-2.5 h on a 3090 (EST).

---

## 6. D2 curriculum: amendment status and the first K3 row

**Already drafted.** `PREREGISTRATION_OMNI_STAGE_D_v1.md` amendment **A2 item 8** (DRAFT, lane d2prep, pin `c24d21e0…` unchanged) does what this task asks for:

- The K3 rows (four passes, 2.0M tokens) are drawn only after the 20M rung, inside the last short rung.
- The hard gate: the K3 cache must be sealed, with every pass's xtok verified, before the first K3 row. If D2 reaches 20M first, it checkpoints and waits.
- λ_K3 0.3, the anchor 0.5 and the aligned positions are unchanged.

d2_plan 1.2.0 already implements this (`K3_DEFER_TOKENS = 20,000,000`, seal marker `$LX/distill/d2/k3.sealed`). **I did not write a second amendment.** A2 and A3 already exist, and a parallel "A2" would collide. Below are two additions for the d2prep lane to fold into A2 before it is finalized. Both are text only, and neither changes losses, weights, token totals or LoRA sites:

> **8a [k3plan].** "Sealed" for item 8 means: four pass directories, each sealed by a tool and card class that `$OC/distill/queues/k3_pass_tools.json` records with a passing gate receipt. These are the B2-proven frozen 1.2.4 (`2aae4dc3…`) on GPU6, or the frozen 1.3.0 (`434ced6d…`) on a class whose `k3-v130-gate` (GPU6) or card-class equivalence verdict (ruling `k3-card-class-equivalence-2026-10-15`) passed. In addition, each pass's xtok must verify against its Q38 cache. Passes from different card classes may be mixed; each records its class.
>
> **8b [k3plan, optional].** If D2 reaches the 20M rung before the K3 seal, it may run the long rung (31 × 32,768, KL to Q38, unchanged) first and then the deferred segment (20M to the end of the short run). If the seal comes first, the order stays short, then long. The long rung's tokens, loss and weights are unchanged. This saves up to 4.2 h (upside) or 4.8 h (planning) of waiting, at the cost of putting the K3-row segment after the 32k rung.

**When D2 reaches its first K3 row** (D2 start = D1 registered + 1 h of P2 = 10-16 02:20Z to 11:20Z; to 20M = 20M / rate × 1.3; EST):

| Engine | Time to 20M | First K3 row |
|---|---|---|
| Upside (streamed bf16, 348 tok/s, GPU0+GPU4) | 20.75 h | **10-16 23:05Z to 10-17 08:05Z** |
| Planning (split cell, 151.24 tok/s, GPU6+GPU4) | 47.75 h | **10-18 02:05Z to 11:05Z** (and GPU6 is D2's, see section 4.4) |

Which K3 option is sealed by that point:

- **Upside:** only O4 seals in time, at 10-16 16:30Z (if the 3090 rate is high) to 10-17 19:05Z. Under O2/O3, D2 waits at 20M.
- **Planning:** O3 (cooled) and O4 both seal in time.

**Required code change (not mine; lane d2prep owns d2_plan):** d2_plan 1.2.0 `k3_state` (`d2_plan.py:165-182`) counts a pass as sealed only if `tool_sha256 == sha256($Q/frozen/k3_teacher_cache-gpu6.py)`. Item 8a needs it to accept the shas and classes listed in `$Q/k3_pass_tools.json` (written by `k3_lib.k3_record_pass`), with each gate receipt checked. Until that change lands, enable neither 1.3.0 nor the 3090 route (install step 4).

---

## 7. GPU6 thermals: numbers and options for Cyril

**Cost of the throttling (EST, sections 3 and 4.4):**

| Tool | Per pass now | Per pass cooled | 4 passes now | 4 passes cooled | Time saved |
|---|---|---|---|---|---|
| 1.3.0 | 18.8-33.6 h | 11.3-14.0 h | 75-134 h | 45-56 h | about 1.3-3.3 days |
| 1.2.4 | 29.8-43.9 h | 15.2-22.0 h | | | |

Both directly reduce the D2 wait.

**Diagnosis:**

- The card is passive: no fan is reported and the board power limit is 250 W.
- It sits at its 82 C max operating temperature with SW thermal slowdown active, at about 95 W. Power is not what limits it; heat removal is.
- It is in a Supermicro X10DRG 4U GPU chassis. On boards of this generation the BMC steers the fans from CPU and system sensors and does not read the NVIDIA GPU temperature, so a passive GPU can cook while the fans idle (common behaviour; the BMC fan mode is UNVERIFIED because `ipmitool` is not installed).

**Options, all Cyril's call. I changed none of them.**

1. **BMC fan mode: Full Speed (or Heavy IO).** In the Supermicro IPMI web UI, Configuration → Fan Mode.
   - Reversible, no reboot, numerics untouched.
   - Cost: noise.
   - This is the first thing to try.
2. **A fixed duty for the GPU fan zone through IPMI raw commands**, if Full is too loud. This needs `sudo apt install ipmitool`, and the exact zone and duty values are board-specific. Check them against Supermicro's documentation for X10DRG-OT+ before use.
3. **Physical airflow.**
   - Air shroud and blanking plates in place, intake unobstructed, filters clean.
   - GPU6 (PCI 88:00.0, the CPU2 side) next to a hot neighbour?
   - GPU5 is never used. Pulling it, or swapping GPU6 into the slot with the strongest airflow, costs nothing in compute.
4. **Room or rack intake temperature.**
5. **Not useful:**
   - The power limit: the card already draws about 95 W against 150-250 W.
   - Clock locking (`nvidia-smi -lgc`): it adds no cooling.
6. **Software only:** move K3 passes to the 3090s (O4). This is staged as `k3_multicard.sh`.

**After any change:** the queue's thermal log (`$Q/logs/gpu6/<job>.thermal.log`, one line per minute) shows the effect directly. Look for `reasons=0x0` under load and SM clocks above 1,200 MHz. The watchdog thresholds (84 C ×3, 85 C hard) stay as they are.

---

## 8. What is staged

Everything is in `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/distill/` and mirrored on the server under `$OC/distill/`. Local and server sha256 match for every file (`$LX/scratch/k3accel/plan/final/sha256.txt`).

| File | Version | sha256 | State |
|---|---|---|---|
| `tools/queue_gpu6.sh` | **1.4.0** | `863f30895361e92e629e4c8d406442b3a05cdfe9ca408b6eb401f1759d8f8dd7` | staged, **not installed** (`$Q/gpu6.sh` is still 1.3.0 `bf22ec8f…`) |
| `tools/k3_lib.sh` | 1.0.0 (new) | `e70ba063b34ccb64b8d85506c80155cbb10922d63c4c72c2bc6556199f07da3d` | staged |
| `tools/k3_multicard.sh` = `queues/k3_multicard.sh` | 1.0.0 (new) | `bd52dcfbb074eb42310fd50661e4eae8738c7a2bc0de480fe73aeac707e7b288` | written to `$Q`, **not started** |
| `tools/test_k3_lib.sh` | | `09b0c68b51304272a91609fabbb431eb990af07d6bcd85aee3c8fef7c0052231` | |
| `tools/test_queue_gpu6.sh` | (+ plan checks for 1.4.0) | `ff3eab30eaa2e06ea82382f4c938fcfe0b457be7b3c827601f4b09a033984e54` | |
| `tools/test_k3_multicard.sh` | | `bdc3089fcc00e7e2ced654eb58f1a0c4ba3af3e79235088ba10396ada20e7d6a` | |
| `tools/k3_schedule_projection.py` | | `af785023dd720f790d07973700108d5d756f3dc9a4ddb0809eb153e94368e5f4` | the arithmetic in sections 3-4 |
| `queues/k3_card_class_ruling.txt` | | `5da95bfb4baa0d6ecb1939b9af6f467d72c5cbc6dd2fe25a92d7e5e02c408e5e` | written to `$Q` |
| `STAGE_D_RULINGS_2026-10-12.md` | + ruling entry | `74ae3b780292f95d33278517a6c708124a0ec859ff45edb9a573b815e7ffe067` | was `aaacf362…` |

The previous tools copies were archived in `tools/_superseded/` on both sides: `queue_gpu6_1.3.2.sh` (`8bb5a5f0…`, lane infra's, never installed) and `test_queue_gpu6_1.3.2.sh` (`005df700…`). 1.4.0 contains all of 1.3.2.

### 8.1 queue_gpu6.sh 1.4.0 vs 1.3.2

- **Shared helpers.**
  - `wait_lexar` through `b2_ok` (1.3.2 lines 199-389) moved byte for byte into `k3_lib.sh`. The builder checked each line.
  - The thermal threshold reads became `k3lib_thermal_init`, so the extra-card runner reuses the same watchdog, RAM ballast and B2 gate code.
  - The queue refuses to start unless it finds `k3_lib.sh 1.0.0`.
- **Must-fix from the review: fixed geometry per pass.**
  - Each pass records `{tool sha, card class, group tokens}` in `<pass dir>.geometry` at its first start, and a resume that differs is refused (rc 65).
  - An out-of-memory error never changes group tokens. With 1.3.0, it lowers `--moe-group-batch` to 1, which is numerics-neutral per k3accel's w4. That is followed by one plain retry, then HALT for a decision.
  - The OOM test now reads only the current attempt's output. 1.3.2's `tail -8` could also match an earlier attempt's OOM lines; the test found this.
- **`k3-v130-gate`.**
  - Runs only with `$Q/k3_v130_review.txt` present (line 1 `k3-v130-review-pass`, plus a line `sha256=434ced6d…`).
  - Freezes 1.3.0 at the pinned sha as `$Q/frozen/k3_teacher_cache-gpu6-v130.py`.
  - Runs the bit-identity windows on GPU6 with the pass flags (section 5, part 1) and writes `$Q/k3_v130_gate/<class>/verdict.json`.
  - PASS: the passes run 1.3.0 with `--no-fadvise-drop` (review item 2), `--kda-graph-keep kda-only` (review item 3: the P40-tested path) and `--moe-group-batch 4` (the w4-proven count; `auto`, about 7, has never run).
  - Anything else: 1.2.4.
- **Pass claims.**
  - GPU6 takes the lowest unsealed pass that nobody else holds, under flock `$Q/k3_passes.lock`.
  - A pass begun by 1.2.4 finishes on 1.2.4.
  - `$Q/k3_gpu6.max_passes` (re-read before every pick) caps how many passes GPU6 starts.
  - When GPU6 has nothing left to run, it moves on to H3 and j6. The route probe runs wherever the last pass seals.
  - Every seal is recorded in `$Q/k3_pass_tools.json`.
- **Adoption.** A `b2-identity` run or a `b2-fp32` pass left behind by an earlier queue is watched to its end, not started again.
- **Card lock: unchanged from 1.3.2.** A `lanes.py hold-lock` flock on `$Q/gpu6.lock` is held while the queue or any `$Q/frozen/` job lives.
- **Not addressed.** Review items 4-6: retrying a refused capture every row, checking only the first replayed shape, and the `dequant_seconds_engine` label. These affect speed and telemetry only, and live in the tool, which I did not change.

### 8.2 k3_multicard.sh 1.0.0 (GPU0 or GPU4)

- **Refusals.** GPU5, GPU6 (the queue's), and the P40s (MLA at T = 4,096 does not fit).
- **Gates, each one waits:**
  1. `$Q/d1.sealed` exists.
  2. `$Q/k3_multicard.go`: line 1 `k3-multicard-go`, plus a line `gpus=` that names this card.
  3. The B2 PASS (the queue's own `b2_ok`).
  4. The ruling file.
  5. `$Q/gpu<N>.lock`, taken through `lanes.py hold-lock`. The helper also matches this card's jobs by their per-card `--state-dir`.
  6. The card-class eqcheck: once per class and tool, behind flock `$Q/k3_cardclass/<class>.lock`. Verdict in `$Q/k3_cardclass/<class>/verdict.json`.
  7. RAM and Lexar free space per pass.
- **How it runs passes.**
  - It claims passes from the top (3, 2, 1, 0) with the same `k3_run_pass` as GPU6.
  - Group tokens are 32,768: the eqcheck's own geometry, fixed per pass.
  - `eqcheck_only=1` in the GO file stops it after the check, so the class can be measured before deciding K3-first or D2-first.

### 8.3 Tests (CPU; they read the GPU names and temperatures only)

The files were staged from these builds; their shas match the tools copies.

**`test_k3_lib.sh`** (stage2 run, lib `e70ba063…`):

```
K3 LIB TEST (k3_lib.sh 1.0.0, e70ba063b34ccb64): 36 passed, 0 failed
```

Covered: the card class; the review and tool choice; pass claims (sealed, live claim, bound geometry, own claim, dead claim, foreign seal); geometry; OOM sequences on a fake tool (1.3.0: batch 4 → 1 → 1, group tokens 65,536 throughout; 1.2.4: HALT after 2); refusal of a changed group size; a failure with stale OOM lines in the log; the tool record; legacy geometry; all-sealed.

**Negative control** (`--negative`: a lib copy that resumes an OOM at 32,768, which is what 1.3.2 does):

```
FAIL oom-v130-gt: got '65536,32768,32768' want '65536,65536,65536'
K3 LIB TEST (k3_lib.sh 1.0.0, d10e88b384c41bc8): 35 passed, 1 failed
```

**`test_k3_multicard.sh`:**

```
K3 MULTICARD TEST (k3_multicard.sh 1.0.0): 16 passed, 0 failed
```

Covered: 4 refusals; 7 gate readers; the plan for GPU0, which resolves to `nvidia-geforce-rtx-3090-sm86`, tool v124 `2aae4dc3`, group tokens 32768, thermal stop 95 C ×3 / hard 96 C; an unknown selftest.

**`test_queue_gpu6.sh` on 1.4.0** (first stage run, lib `f541674e…`):

```
QUEUE SELFTEST (queue_gpu6.sh 1.4.0): 30 passed, 1 failed
MISSING: ST ballast: first=3 max=3 last=0 released=1 job_oom_adj=500
```

- The thermal, b2ok, lanes, bogus and the 3 new plan checks all passed.
- **The ballast case is flaky on both versions.** Four alternating reruns produced:
  - 1.4.0: `last=2`, `last=0`, `last=3`, `last=2` (1 of 4 pass).
  - 1.3.2 (the file now in `_superseded/`): `last=0`, `last=0`, `last=2`, `last=3` (2 of 4 pass).
  - Log: `$LX/scratch/k3accel/plan/stage/ballast_flake.log`.
- The cause: swap is full, so the dummy job's RssAnon drops while it sleeps, and the ballast re-grabs memory. `ram_ballast` is byte-identical in 1.3.2 and 1.4.0.
- **Final run on the staged tools copies** (`$LX/scratch/k3accel/plan/final/`, 02:00Z-02:12Z):

  ```
  K3 LIB TEST (k3_lib.sh 1.0.0, e70ba063b34ccb64): 36 passed, 0 failed
  K3 LIB TEST (k3_lib.sh 1.0.0, d10e88b384c41bc8): 35 passed, 1 failed      <- --negative, as designed
  K3 MULTICARD TEST (k3_multicard.sh 1.0.0): 16 passed, 0 failed
  QUEUE SELFTEST (queue_gpu6.sh 1.4.0): 31 passed, 0 failed                 <- the ballast case passed this time
  ```

---

## 9. Install steps (in order; none done)

1. **Now, any time:** Cyril's cooling change (section 7). It is independent of everything else.
2. **Install queue 1.4.0 while the identity gate or the fp32 pass is still running** (before about 10-15 12:30Z for identity). 1.4.0 adopts both.

   ```bash
   Q=/home/cereal/omni-campaign/distill/queues; T=/home/cereal/omni-campaign/distill/tools
   bash $T/test_k3_lib.sh && bash $T/test_queue_gpu6.sh $T/queue_gpu6.sh   # expect 36/0; 31/0 (ballast may flake, see 8.3)
   ps -o pid,ppid,etime,args --ppid 107948                                  # at 02:04Z: 306480 = the python job (KEEP), 306481 = the watcher subshell
   cp -p $Q/gpu6.sh $Q/gpu6.sh.1.3.0-live-$(date -u +%Y%m%dT%H%M%SZ)
   kill -TERM 107948 <watcher subshell pid>                                 # NOT the python identity/fp32 job
   cp $T/queue_gpu6.sh $Q/gpu6.sh
   START_AT=b2 setsid nohup bash $Q/gpu6.sh > $Q/logs/gpu6/queue-1.4.0.log 2>&1 < /dev/null &
   ```

   Expect these lines:
   - in `queue-1.4.0.log`: `card lock … joined behind orphaned helper(s) 726165`;
   - in `gpu6.status`: `b2-identity START "ADOPT running pid; …"` (or `b2-fp32` adopted).

   Check with `python3 $T/lanes.py show --lane L-K3`.
3. **Optional:** to give 1.3.0's pinned layer arena room, start the queue with `K3_OVERHEAD_GIB=23` (section 2.4).
4. **Enable 1.3.0, only after d2_plan accepts `$Q/k3_pass_tools.json` (section 6, lane d2prep).**
   - Write `$Q/k3_v130_review.txt`:

     ```
     k3-v130-review-pass
     sha256=434ced6d7e8e22e4179bb2332d4695803dd89a38407805126261500c112fa3c7
     by: <reviewer>, 2026-10-15: no numerics defect; review item 1 handled by queue_gpu6 1.4.0
     ```

   - The file is read when the queue starts. If the queue is already running, restart it the same way as step 2. It resumes or skips whatever is done.
   - The gate runs after B2 PASS.
5. **At D1 seal**, in this order:
   - P2 on GPU0.
   - `printf 'k3-multicard-go\ngpus=0\neqcheck_only=1\n' > $Q/k3_multicard.go`, then `setsid nohup bash $Q/k3_multicard.sh 0 > $Q/logs/k3mc-gpu0.runner.log 2>&1 < /dev/null &`.
   - Apply the rule in section 4.5.
   - For K3-first: rewrite the GO file as `gpus=0,4` (without `eqcheck_only`), `echo 1 > $Q/k3_gpu6.max_passes`, and start runners for 0 and 4.
   - For D2-first: delete the GO file and let D2 take GPU0/GPU4.
6. **Planning engine instead:** as in section 4.4. Set `$Q/k3_gpu6.max_passes` to the passes GPU6 has already started, and run `k3_multicard.sh 0` with `gpus=0`.

Until step 2 is done, the live 1.3.0 queue still contains the out-of-memory group-size switch (`queue_gpu6.sh` 1.3.0 `GT=32768`). A K3 pass that runs out of memory under 1.3.0 would mix geometries.

---

## 10. What I could not verify, and risks

- **Nothing new ran on a GPU.** Untested so far:
  - the GPU6 1.3.0 gate (and therefore 1.3.0 on torch 2.11 / sm_75);
  - the 3090 eqcheck;
  - k3_multicard's pass path;
  - 1.4.0's pass loop, gate and adoption on real jobs.

  The pass logic is tested on a fake tool; the gate and eqcheck command lines are not executed by any test.
- **The rate of a 3090 pass** (4.2-14 h) is a range from spec sheets and the P40 floor. Its memory fit at T = 4,096 MLA is assumed (fused SDPA on sm_86).
- **The cooled GPU6 per-pass time** rests on one cool layer (L01) and the assumption that the card is at least P40-fast.
- **All dates are EST.** That includes the D1 pace (7 layers sampled), the identity pace, the fp32 duration (unmeasured on sm_75) and the D2 design rates (P2 not run).
- **The BMC fan mode and chassis fan speeds are unread**: no `ipmitool`, root-only `/dev/ipmi0`.
- **d2_plan does not yet accept 1.3.0 or 3090-sealed passes.** Without its change, step 4 must wait. xtok run by the frozen 1.2.4 on a 1.3.0-written cache is assumed compatible (NUMERICS_VERSION and the layout are unchanged): UNVERIFIED.
- **RAM is the binding constraint.** Swap is full now. Two K3 passes at once leave about 40 GiB, and K3 plus D2 upside leaves about 40 GiB. The gates wait rather than kill, but anything else large started by another lane delays K3.
- **The ballast selftest flakes** on 1.3.2 and 1.4.0 alike under the current swap pressure (section 8.3).

## Receipts

- `$LX/scratch/k3accel/plan/receipts/`: `clockfit.py`/`.json` (`1479c582…`/`aa5a9175…`), `k3_schedule_projection.py`/`.json`/`.txt` (`af785023…`/`28f730e3…`/`b3423521…`), `gpu6logs-20261015T0120Z.tgz` (`024b93d2…`).
- `$LX/scratch/k3accel/plan/stage{,2}/`, `final/`: the test runs.
- Thermal and pass logs: `$Q/logs/gpu6/{b2_pass,b2.thermal,b2-identity_pass,b2-identity.thermal}.log`.
- P40 1.3.0: `$LX/scratch/k3accel/runs/v130-20261014T200903Z/`, `…/v130-20261014T231558Z/`.

`$LX` = `/mnt/lexar/omni-campaign`; `$Q` = `$OC/distill/queues`.
