# Strata Q4 (IQ4_XS): identity, download, and a head-to-head plan against the served IQ3_S

> [Cyril quote withheld]

Date: 2026-10-21 (lane q4). Server copy: `/home/cereal/ai-serve/STRATA_Q4_PLAN_2026-10-21.md`.
Evidence: `/home/cereal/ai-serve/evidence/q4-20261021/` on the AI server.
Code: `C:/Users/cereal/Documents/ai training/ai-serve/strata/` (`gguf_ident.py`, `strata_q4_download.sh`, `q4_swap/`), mirrored to
`/home/cereal/ai-serve/strata/`.

**Nothing in this plan touches GPU0, GPU4 or GPU6 until Cyril approves the swap (section 5).** The download and the
identity proof ran on the CPU and disk only; serving was never stopped.

## 1. Where the Q4 build comes from

- **The served file's own repo has no Q4.** `jimmy31chen/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata`
  (commit `17876139`) publishes one GGUF, `RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf` (sha `cb5958e5…`, the file we
  serve), plus a Strata pack. Its card says the file is `llama-gguf-split --merge` of the 8-shard IQ3_S in
  `0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-GGUF`, and names
  `base_model: 0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored`.
- **The upstream repo has the Q4 builds.** `0bserverx/…-GGUF` at commit `75358160` publishes 18 families, 8 shards each.
  Five IQ families were made with the same V6 importance matrix.

  | Family | Bytes | imatrix | Use here |
  |---|---:|---|---|
  | IQ3_S (served, merged by jimmy31chen) | 88,882,766,080 | V6 | live |
  | **IQ4_XS** | **97,473,155,840** | V6 | **downloaded** |
  | Q4_0 | 99,959,519,712 | no | no: larger, no imatrix |
  | IQ4_NL | 100,079,450,880 | V6 | no: larger than IQ4_XS, same 4-bit grid class |
  | Q4_K_M | 119,150,723,552 | no | rejected in the design (15-17 expert layers on CPU) |

  IQ4_XS is the design's gated phase-2 quant.

## 2. Identity: is the IQ4_XS the same model as the served IQ3_S?

Tool: `gguf_ident.py` 1.0.0 (stdlib only). It reads GGUF headers from local files or over HTTP Range requests (only the
header bytes; 16 MB for shard 1, 8 MB for each other shard), treats an 8-shard family as one model, and compares field
by field.

**Step 1: header proof, made BEFORE the download** (`compare_local_merged_vs_hf_IQ3_S.json`,
`compare_local_merged_vs_hf_IQ4_XS.json`).

The served merged file against the upstream IQ3_S shards:
- Every one of the 1,224 tensors has the same type, shape and byte size.
- No general, quantize or tokenizer field differs.
- The merged file carries `split.count = 0`, which is the `llama-gguf-split --merge` marker.

So the served file is the upstream IQ3_S, merged, as its card says.

The served merged file against the upstream IQ4_XS:

| Check | Result |
|---|---|
| `general.architecture` | equal (`qwen4exp`) |
| hparams (`qwen4exp.*`: 48 blocks, 512 experts / 10 used, width 2560, SSM, hyper-connections, PLE n-gram table, 262,144 context) | all 36 equal |
| tokenizer (`tokenizer.ggml.*` + chat template, sha256 of each value) | all 9 equal; combined `ec02f85c5884d0cc…` |
| identity fields | equal: `general.name` "Rvn Qwen38 Flash Next v6 F16 Direct Hf", basename `rvn-qwen38-flash-next-v6`, finetune `direct-hf`, size label `512x56B`, base model Qwen3.8 Flash Next |
| `quantize.imatrix.*` | equal: file `imatrix-v6-130.dat`, 902 entries, 130 chunks, same dataset path |
| tensor names / shapes | 1,224 = 1,224, all shapes equal |
| only differences | `general.file_type` 26 -> 30; 568 tensors IQ3_S -> IQ4_XS (expert gate/up, shared-expert gate/up, SSM/hyper-connection projections, `token_embd`), 48 `attn_qkv` Q4_K -> Q5_K |
| tensors that kept their type | 608 tensors, 52,495,495,680 B: the per-layer n-gram table (IQ4_NL, 28.8 GB), all 48 expert down projections (IQ4_NL), norms (F32), F16 tensors, `output.weight` (Q6_K) |

Upstream shard 2 of every IQ family (one tensor, `per_layer_token_embd.weight`) has the same LFS sha256 (`1723660f…`).

**Step 2: byte proof, after the download** (`tensor_hash_IQ3_S_sametype.json`, `tensor_hash_IQ4_XS_sametype.json`,
`identity_same_bytes.json`, summary in `q4_receipt.json`).
- The data bytes of all 608 same-type tensors are hashed in the served IQ3_S file and in the downloaded IQ4_XS.
- Two quantizations of one checkpoint with one imatrix must agree byte for byte there. Two different checkpoints
  cannot share 52.5 GB of quantized bytes by chance.
- Result: ****608 of 608 tensors byte-identical (52,495,495,680 B), 0 differ, 0 missing** (2026-10-22 00:46Z; the IQ4_XS side was hashed in 470 s, the IQ3_S side in 302 s). An early, separate hash of the 28,800,138,240 B n-gram table alone gave the same sha256 in both files, `dd55c28902f38cd8…` (`early_ple_tensor_identity.json`). **The IQ4_XS is the same checkpoint as the served IQ3_S, quantized differently.****

**Tokenizer used by the A/B.** The omni_lab `gguf` engine renders prompts with `toolcall_selfgen`'s pinned tokenizer
(`/home/cereal/models/Qwen3.8-27B-Uncensored/tokenizer.json`, sha `06b95093…`) and sends token ids.
- Checked against the served GGUF: all 248,077 ids map to the same token strings.
- The GGUF adds 243 `[PAD…]` entries only.
- Both quants share that tokenizer (hash equal above).

## 3. Download

- **Script:** `strata_q4_download.sh` (1.0.2), detached, logging to `evidence/q4-20261021/download.log`.
- **Target:** `/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata-IQ4_XS/` (root disk).
- **Pinning:** the file list is pinned to commit `75358160`. The script refuses if the pinned tree's sizes or LFS sha256s
  differ from its own list.
- **Free space:** checked before fetching, so `/` keeps at least 100 GB after the download.
- **Per shard:** curl resumes into a `.part` file, the size is checked, then sha256 must equal the LFS oid before the
  file is renamed. Then `<shard>.sha256.ok` is written, and `SHA256SUMS` once all 8 are done.
- **Identity:** the step-2 proof runs on the local bytes.
- **Rate:** capped at 50 MB/s in total. The 1 GbE NIC (`enp129s0f0`, 1000 Mb/s) also carries LAN serving.
- **What happened on the way:**
  - The CDN cancels long HTTP/2 streams (curl rc 92 about every 6 min on the 28.8 GB shard 2). 1.0.1 switched to
    HTTP/1.1 and retries while each attempt adds bytes.
  - One connection then got 6.6 MB/s, and four connections got 14.4 MB/s in total (about 3.6 MB/s each). So the job
    now fetches all remaining shards in parallel (`Q4_PAR=7`).
- **omni_lab runs (kind `build`, lane L-SERVE):**
  - 38053, 38054, 38056 and 38057: stopped by hand at each restart, each with a note.
  - 38065 (`pass`, 23:38Z-00:46Z, receipt `q4_receipt.json`): the run that finished.
- **Result:** **done.** All 8 shards were verified against their LFS sha256 (list in `/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata-IQ4_XS/SHA256SUMS`). The fetch phase of the final run took 3,330 s. `/` had 271,971,102,720 B free afterwards (floor 100 GB). The download used no GPU, and serving stayed up throughout: strata-llama, strata-gate, gen-worker-text and media-gate were all active at the end, `/healthz` was ok, and GPU0/GPU6 still used 22,428 / 40,722 MiB.

## 4. VRAM fit on GPU0 + GPU6 at 6 slots x 32k (+ RAM)

**Method.** `gguf_ident.py place` applies llama.cpp's layer-split rule (`load_tensors`):
- layer `il` goes to `upper_bound(cumsum(ts)/sum, il/49)`, with the output layer as index 48;
- `-ncmoe N` keeps the experts of layers 0..N-1 on the CPU;
- `token_embd` and `per_layer_token_embd` stay in host RAM.

**Calibration on the live IQ3_S.** nvidia-smi read 22:4xZ: GPU0 22,428 MiB used of 24,576, GPU6 40,722 of 46,080.
- `-ts 16,32` puts layers 0-16 on GPU0 (20,030 MiB of weights) and 17-47 plus output on GPU6 (36,998 MiB).
- The design's KV (q8_0, 196,608 tokens: 205 MiB per full-attention layer) and SSM state (18.7 MiB per recurrent layer)
  account for 1,062 / 2,068 MiB.
- The rest, compute buffers plus CUDA context, is 1,336 / 1,656 MiB. That residual is held fixed below.
- Host weights: 27,726 MiB.

**The IQ4_XS is exactly 8,192 MiB bigger, almost all of it in GPU-resident layers** (1,343-1,365 MiB per layer against
1,175-1,192).

| Placement (estimate, not measured) | GPU0 free | GPU6 free | Host weights |
|---|---:|---:|---:|
| IQ3_S live, `-ts 16,32` | 2,148 MiB (measured) | 5,358 MiB (measured) | 27.1 GiB |
| IQ4_XS, `-ts 16,32`, all on GPU | **-734** (does not fit) | 109 | 27.1 GiB |
| **IQ4_XS, `-ts 17,31 -ncmoe 3`** (staged) | 1,799 | 1,476 | 30.9 GiB |
| IQ4_XS, `-ts 18,30 -ncmoe 4` (fallback) | 1,733 | 2,842 | 32.2 GiB |
| IQ4_XS, `-ts 19,29 -ncmoe 5` | 1,485 | 4,390 | 33.5 GiB |

The design estimated `-ncmoe 8`; three expert layers are enough by this count. Host RSS should rise from about 31 GB
to about 35 GB; MemAvailable was 127-133 GiB today. GPU6 drops from 31 layers to 30, but each is about 14 % heavier.

**Cost of the 3 CPU expert layers (estimate).**
- **Decode:** 10 of 512 experts per token is 25.4 MiB per layer, so about 76 MiB of host reads per token per stream.
  That is about 1.5-2 ms on this host, against about 21 ms per token today (47.9 tok/s).
- **Prefill:** `--op-offload` (on by default) copies those 3 x 1,300 MiB over PCIe 3.0 for every 512-token ubatch.
  That is roughly 0.3 s per ubatch, so about +5 s on an 8.5k-token Sprite prompt (13.4 s today).
- **GPU weight reads per token** grow about 15-20 %.

**Expected, to be measured in the swap window:**
- single stream about 38-43 tok/s (today 47.9);
- 6-slot aggregate about 75-90 (today 100.3);
- Sprite-shaped first token about 18-21 s (today 15.1 s p50).

These are estimates and nothing more. If prefill is the problem, `--no-op-offload` (CPU prefill for those 3 layers) is
the variant to measure.

## 5. Head-to-head plan (prepared, NOT executed)

### Phase A: quality A/B on P40s (no serving card, no Sprite impact)

The two quants run on the same hardware with identical flags, so only the quant differs.

- **Cards.** Preferred: 4 x P40 (GPU1, 2, 3, 5) once the L-EXP arch-v2 probe on GPU3/5 ends.
  - `-ts 12,12,12,12`, everything on GPU.
  - Predicted free: IQ4_XS 5.3-7.5 GiB per card, IQ3_S 7.5-9.4 GiB.
- **Fallback.** GPU1 + GPU2 only (L-TRAIN, idle, borrowed):
  - `-ts 32,16 -ncmoe 18` for both arms;
  - predicted free: IQ4_XS 1.3 / 1.6 GiB;
  - 50 GiB of experts in host RAM and much slower prefill: about 2-3x the wall time.
- **Server per arm** (by hand, one arm at a time, under the lane locks):
  ```
  OC=/home/cereal/omni-campaign; L=$OC/export/llamacpp/bin
  python3 $OC/distill/tools/lanes.py claim-run --lane L-EXP --agent claude --gpus 1,2,3,5 --borrow "q4 A/B; L-TRAIN queue empty" \
    --job strata-q4-ab-<arm> --log /home/cereal/ai-serve/lane-logs/strata-q4-ab-<arm>.log --omni-lab-kind eval -- \
    env LD_LIBRARY_PATH=$L CUDA_DEVICE_ORDER=PCI_BUS_ID nice -n 10 $L/llama-server -m <GGUF> --host 127.0.0.1 --port 18411 \
      -ngl 999 -sm layer -ts 12,12,12,12 -fit off -np 1 -c 32768 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -t 8 \
      --no-mmap --lazy-mode off --jinja --no-webui --seed 1234
  ```
  - `<GGUF>` is the served merged IQ3_S, or IQ4_XS shard 1 (llama.cpp opens shards 2-8 itself).
  - `claim-run` exports `CUDA_VISIBLE_DEVICES` = the claimed cards' UUIDs (`lanes.py` 1143-1144; GPU1 `c95fb5b3`,
    GPU2 `a92b0dfe`, GPU3 `08b615f9`, GPU5 `ce4a1da0`). Never start this server outside `claim-run`: with every card
    visible, `-ts 12,12,12,12` would land on GPU0-3, and GPU0 is a serving card.
  - `-np 1` keeps greedy decoding batch-independent. `-t 8` and `nice` keep CPU away from strata-llama (`-t 16 -tb 32`).
  - RAM: about 31 GB more resident for the n-gram table. Check MemAvailable stays at or above 60 GiB before each start.
- **Runner** (`/home/cereal/omni-campaign/lab`, intake venv). The `gguf` engine has never run end to end, so run
  `--dry-run` first, then a 3-case smoke.
  ```
  PY=/home/cereal/.venvs/chimera-intake/bin/python; TOK=/home/cereal/models/Qwen3.8-27B-Uncensored; U=http://127.0.0.1:18411
  $PY omni_lab.py run --model <GGUF> --engine gguf --server-url $U --tokenizer-dir $TOK --suite toolcall_v1 --limit 3 --cap 1024 --dry-run
  $PY omni_lab.py run --model <GGUF> --engine gguf --server-url $U --tokenizer-dir $TOK --suite toolcall_v1 --cap 1024 --seed 1234 --save-test strata-q4-toolcall --baseline none
  $PY omni_lab.py run --model <GGUF> --engine gguf --server-url $U --tokenizer-dir $TOK --suite codex_recovery_bfcl --cap 1024 --seed 1234 --save-test strata-q4-bfcl62 --baseline none
  $PY omni_lab.py run --model <GGUF> --engine gguf --server-url $U --tokenizer-dir $TOK --suite bank_v1 --cap 4096 --seed 1234 --save-test strata-q4-bankv1 --baseline none
  # second arm: the same three with --test strata-q4-toolcall|strata-q4-bfcl62|strata-q4-bankv1 --model <other GGUF>
  ```
- **Order of arms.** IQ3_S, then IQ4_XS, then IQ3_S again on `toolcall_v1` + `codex_recovery_bfcl` only. The repeat is
  the determinism control; any case that flips between the two IQ3_S runs is noise and is excluded from the decision.
- **Suites:**
  - `toolcall_v1@1`: 185 cases over 8 lanes, machine-graded by `toolcall_selfgen.grade_case`. Its cases and held-out
    function names are in every corpus screen.
  - `codex_recovery_bfcl@oct13_v3`: the 62 BFCL held-out tasks, graded by `chimera_recovery.evaluate`.
  - `bank_v1@1`: 16 cases, reader-graded only.
- **Grading `bank_v1`.** Blind: responses are exported with the arm labels shuffled, graded, then loaded with
  `omni_lab.py grade <run> --file grades.json --grader human:claude`. Cyril's own grades (`human:cyril`) outrank them.
- **Compare:** `omni_lab.py compare <iq3s run> <iq4xs run> <iq3s repeat>`.

**Decision rule (proposed; Cyril to confirm before Phase A):**
- adopt IQ4_XS only if, on `toolcall_v1` + BFCL-62 (247 paired cases, noise cases removed), it wins the discordant
  pairs with an exact two-sided McNemar p <= 0.05;
- AND it has no more bank_v1 cases graded worse than better;
- AND Phase B's measured serving numbers stay inside the limits there.

A tie means IQ3_S stays: it is faster, already fits, and is a known quantity.

- **Wall time:** not measured. Prompt length and output caps dominate, so read it off the 3-case smoke before
  committing the cards.
- **What the A/B does not do:** its outputs are eval rows in omni_lab only and never go to the corpus or the intake.

### Phase B: serving fit and speed on GPU0 + GPU6 (needs Cyril's approval: it takes the serving cards)

Staged and not deployed:
- `q4_swap/strata_llama.sh` (1.1.0). With no drop-in it is the same file, checks and command as the live 1.0.0. With
  `STRATA_MODEL=iq4xs` it accepts the IQ4_XS only when all 8 shards have their size and LFS sha receipt AND
  `q4_receipt.json` has `byte_identity.all_equal == true`.
  - Tested locally on fake shards: 7 cases (ok; missing receipt; receipt false; receipt not literal `true`; wrong shard
    sha; short shard; unknown model). Each refusal exits 2.
- `q4_swap/q4-iq4xs.conf`: a systemd drop-in with `STRATA_MODEL=iq4xs`, `STRATA_TS=17,31` and
  `STRATA_EXTRA=--no-mmap --lazy-mode off -ncmoe 3`.

**Exact procedure** (on the server; `$S=/home/cereal/ai-serve/strata`):
1. **Preconditions.**
   - `q4_receipt.json` says `all_equal: true`.
   - Phase A met the decision rule, or Cyril asks for the measurement anyway.
   - Cyril's approval, quoted verbatim in the claim.
2. **Quiet window.** No sprite or cyril Strata request for 10 minutes. `e2e_serving.py quiet` checks the sprite tier only,
   so use this query (read-only role):
   ```
   SELECT tier, round(extract(epoch FROM now() - max(received_at))) AS idle_s FROM request_log
    WHERE service = 'strata' AND tier IN ('sprite', 'cyril') GROUP BY tier;      -- both >= 600
   curl -s -H "Authorization: Bearer <strata_key_cyril>" http://127.0.0.1:18400/gate/status   # no active/waiting sprite or cyril
   ```
3. **Pre-warm the page cache** (no downtime): `nice -n 15 ionice -c3 cat /home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata-IQ4_XS/*.gguf > /dev/null`.
4. **Stop corpus generation:** `sudo systemctl stop gen-worker-text`. It stops within 120 s, and the batch resumes from its
   journal. This step is required: `gen_worker_text.py` hard-codes the IQ3_S sha in every row's provenance, so it must
   not write while the IQ4_XS serves.
5. **Deploy the launcher** (no behaviour change yet):
   `install -m 755 $S/q4_swap/strata_llama.sh $S/strata_llama.sh && bash -n $S/strata_llama.sh && sha256sum $S/strata_llama.sh`.
6. **Select the IQ4_XS:**
   `sudo install -D -m 644 $S/q4_swap/q4-iq4xs.conf /etc/systemd/system/strata-llama.service.d/q4-iq4xs.conf && sudo systemctl daemon-reload`.
7. **Swap; downtime starts:** `sudo systemctl restart strata-llama`.
   - strata-gate stays up, so Sprite gets an immediate 503 `starting`, not a refused connection.
   - `ExecStartPost` waits for `/health`.
8. **Health checks, all required:**
   - `systemctl is-active strata-llama`;
   - `journalctl -u strata-llama -n 50` shows `start: model iq4xs` and `n_ctx_slot = 32768`;
   - `nvidia-smi --query-gpu=index,memory.used --format=csv -i 0,6` (each card at least 512 MiB free);
   - `curl -s http://127.0.0.1:18400/healthz`;
   - one cyril-tier chat ("hello", 32 tokens).

   If a card runs out of memory at load, the unit loops every 20 s. Then either edit the drop-in to the fallback
   (`STRATA_TS=18,30`, `-ncmoe 4`) and go back to step 6, or roll back (step 10).
9. **Measure** from the workstation with `ai-serve/e2e/e2e_serving.py`. Each test waits for Sprite quiet itself.
   - `strata-load --concurrency 1,6`
   - `strata-prefill --words 1500,4500,7000`
   - `strata-sustain --streams 6 --minutes 12`: GPU6 thermal, against the 81-82 C IQ3_S baseline.

   About 30 min plus the quiet waits. Log the window as one omni_lab `probe` run on L-SERVE with the receipts.
10. **Decide (Cyril).**
    - **Roll back:**
      `sudo rm /etc/systemd/system/strata-llama.service.d/q4-iq4xs.conf && sudo systemctl daemon-reload && sudo systemctl restart strata-llama`.
      Repeat the step-8 checks (expect `model iq3s`), then `sudo systemctl start gen-worker-text`.
    - **Keep:** before `gen-worker-text` starts again, change the IQ3_S identity to the IQ4_XS identity in:
      - `gen_worker_text.py` (`GGUF`, `GGUF_SHA`, `MODEL_ID` and the licence/notes strings, lines 59-61, 237, 489-491);
      - `strata_gate.py` (omni_lab model label, line 627).

      Then run the unit tests, deploy hash-matched, restart strata-gate in a quiet window, and update the runbook and
      the design.

**Sprite downtime estimate:**
- **Per model switch:** about 2.5-3 min of 503 `starting`. Measured today for the IQ3_S: 152 s for a clean restart,
  270 s after a SIGKILL. The IQ4_XS is 9.7 % more bytes, so about 167 s with a warm page cache; a cold cache adds disk
  read time. The 270 s SIGKILL case is the worst case seen.
- **Measure-then-roll-back window:** two switches, so about 5-6 min of `starting` in total. Between them, Sprite is
  served by the IQ4_XS while the benchmarks run, and the benchmarks yield to Sprite.
- **Measure-then-keep:** one switch, about 3 min, plus one strata-gate restart (seconds) for the label.

## 6. Not done here

- The swap (Phase B) and the A/B runs (Phase A): prepared only.
- The omni_lab `gguf` engine's first end-to-end run: it is Phase A's smoke step.
- Any change to `gen_worker_text.py` / `strata_gate.py` provenance strings: only needed if the IQ4_XS is kept.
- MTP speculative decoding and the other quant families.
