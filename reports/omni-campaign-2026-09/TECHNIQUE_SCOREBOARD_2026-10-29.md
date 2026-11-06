# Proven-technique scoreboard for the v2 fold pass

Built 2026-10-29 (02:05Z) from `FOLDING_RECORD_2026-10-22.md` (the record; 524 lines, verification C01-C13) and the omni_lab DB read as `omni_lab_ro` through `omni_lab.py sql` (ingest batch 274). The K3 rows were also re-read from the live receipt on the AI server (read-only, section 9). Nothing was folded, trained, written to omni_lab or deleted to make this file. Paths use the record's short forms: `NF`, `OCL`, `MX`, `MEM`, `REV`.

**How to read every number below**
- **Reader.** "Claude" means a Claude agent read the outputs. "Machine" means a scripted grader or gate. "Codex" means a Codex agent. Cyril holds 11 `scorer='cyril'` rows, all in the August matrix campaign, and none carries a count he produced (record §9). **No Sept-Oct child and no child of Qwen3.5, Q38-27BU or K3 has been read by Cyril.**
- **x/15** is the frozen 15-prompt review. Counts are upper bounds: re-read with three restatements per lane, the LFM2 parent goes 12 → 6, `ARCH.B75` 11 → 2, `ARCH.V19X13G03` 11 → 2 and `ARCH.CL1792M` 8 → 0 (`MX33.F1`).
- **Ratios.** Matrix "4x" cells are 4x for one component. The child is parent-shaped: `COMP.MA.MXK14` stores 0.994287x of the parent (`MX14.F1`). Qwen3.5-4B ratios are given on the safetensors-header count.
- **Cyril's grading rule.** Pass/fail is intelligence retention judged on behaviour. Top-n (10+) token matching is an optimizer target and a metric, never a gate (D 821, ev 1246301). Under his "[Cyril quote withheld]" rule, an arm dropped on a fit number, a PPL/choice gate or an acceptance gate counts as **unreviewed**, not failed. Section 11 counts those arms.
- **"Proven on behaviour"** below means a reader (Claude or Codex) looked at generated outputs and found the child alive. Machine-only counts are listed as **machine upper bound**. No row in this file is a Cyril pass.

---

## 0. Bottom line

1. **Proven on behaviour, at real per-component compression, all on LFM2.5-230M and all read by Claude:**
   - single-site 4x component folds (T8, T5 and T1, plus T3/T4 on conv), 10-12/15 against the parent's 12/15;
   - two-component children at about 0.99x stored (`MXK14` 13/15);
   - removing one block at the right site (`ARCH.X13A` 13/15, 0.947x);
   - a vocab row coreset (`ARCH.V19` 14/15 at 0.795x; `ARCH.V12` 11/15 at 0.763x);
   - vocab plus one excision (`ARCH.V20X13` 10/15 at 0.747x).
2. **Machine upper bounds on bigger parents:**
   - Qwen3.5-4B, one layer excised: 10-15/15 (no reader).
   - Qwen3.5-4B, MLP coreset 4x on all layers: 5/15.
   - Q38-27BU, MLP coreset 4x on all 64 layers: 8/15 (Codex sealed, column B 1).
   - Qwen3.5-0.8B, MLP coreset 4x: 0/162 content, mostly artifact misses.
   - K3 all-experts dense pass: never decoded.
3. **None of the three methods named in the 10-28 plan (ev 1270813: "layer, width and residual-attention folding, K3 first") is proven at the ratios a 2B/4B trunk needs.**
   - **Layer:** proven for one block or layer. The v2 maps are 92:24 (K3) and 64:24 (Q38), and nothing on record survives more than two excisions or 5.28% merge shrink.
   - **Width:** MLP intermediate width is proven per site. Hidden width is alive only down to 0.75 of width on LFM2 (W768, 3/15). v2 needs 0.29 (K3 7168 → 2048) or 0.40 (Q38 5120 → 2048).
   - **Residual attention:** one gradient-trained repair at one site on Qwen3-4B (0.08% of parameters). There is no zero-gradient fold and none at compression.
   - **K3:** dense-pass mechanics on all 92 MoE layers with the folded prefix. No decoded text and no behaviour read.
4. **Composition:** the first composition step is survivable and the second is fatal (`P1.additive-interference`). Width plus any second axis kills (`MX19.F6`, `w36.width-family-kills`). A v2 trunk needs width, depth, heads, vocab and experts → dense stacked together, and no child on record has survived that.
5. **Top-n has never been an optimizer target in any fold** (`gap2:realign-no-topk-fit-objective`; X2 designed, never run). The v2 pass would be the first.

---

## 1. Scoreboard

| Axis | Proven on behaviour (arm, ratio, parent, reader) | Numbers / mechanics only | Failed on a behaviour read | Dropped on numbers = unreviewed (DB, §11) | Families reached |
|---|---|---|---|---|---|
| **Depth excision** | `ARCH.X13A` block 13, 13/15 at 0.947x, LFM2, Claude. `X07` 11/15, `X09` 10/15. Pair `P1301` 7/15 (0.895x) | Qwen3.5-4B X03/X08/X15/X16 15/15, X31 12/15, X00 10/15 at 0.950-0.951x, **machine only, "Nobody has read this child"**; children deleted by ~10-21 | LFM2 `X00`, `X12`, attention 2 each 0/15 alone; 3 blocks 3/15; 4 blocks 0/15 (`ARCH.Q01050913`); X05A 6/15 after re-solve; X05T 0/15 | 38 of 103 dropped arms with an excision component | LFM2 (read) → Qwen3.5-4B (machine). None on Q38, K3 or media |
| **Layer merge** | none read by an agent. Qwen3-0.6B 13+14 at 2.64-2.85% shrink, **machine choice gate** 0.75-1.00; 5.28% on the LFM2.5-transplant host (machine 0.75) | Qwen3-4B → 135M donor-layer groups (~25-30x) partial only **after recovery training** (mixed/machine 3-4/6) | v2 proxy period merge 4:1 from T, ~137x: Claude 0/9 ×3 arms, every continuation an immediate `<|im_end|>` (`V2FOLD.proxy.A/B/C`); SmolLM2 13+14 ~2.63% "repeated sentence/code patterns" | 90 of 197 | Qwen3-0.6B (machine); T line retired |
| **Depth codebook** (k of 36 medoids) | none read | Qwen3-4B p35 400/400 behaviour, 633/1000 generation at **1.0067x** (mixed); p34 296/388 at 1.0134x; ≤34 0/1000 | — | in "attn_qkvo/embed" counts | Qwen3-4B only; no compression |
| **Hidden width** (reader-moment basis) | `ARCH.W896` 4/15 at 0.859x; `W768` 3/15 at 0.723x (both Claude, LFM2). Basis choice proven: the pooled reader moment's top eigenvectors are the only basis that produces a model (`ST.subspace-point`) | read energy below ~0.90 means dead, above it does not predict (`ST.ceiling`). Qwen3.5-4B: 1 hidden-width arm, not alive. SmolLM2 576 → 432 charts on 2 boundaries only (`KS2.CW.576to432`) | `W736` 1/15; W720/W704/W640/W512 0/15; every composed W896 child 0-2/15 | 117 of 278 | LFM2 only |
| **MLP width, per-site** | `C13.T8.S4.L05` mlp_down 12/15; `C11.T1.S4.L00` 10/15; `CL1792` 7/15, `CL1792M` 8/15 at 0.856x (global intermediate, Claude). MLP beats hidden width by 3 cases at matched ratio (`FR3.MLP-beats-width`) | LFM2 ff1280 (2x every site) "~5/15 degraded, not degenerate"; H3 L25 MLP 4x relL2 0.3034 vs prior 0.3266 (`C16.H3.L25.v2`); Astra LFM2 FFN 4x 78/128 vs parent 80 (machine) | all-site: 14 SwiGLU coresets each in the alive band compose to **0/15** (`MX14.F3`); global intermediate ≤640 dead (`MX.ffn-cliff`); `C13.T3.S4.L05` 7/15 | 410 of 627 | LFM2 (read) → Qwen3.5-4B 4x 5/15, 8x 1/15 (machine) → Q38-27BU 4x 8/15 B 1 (Codex) → Qwen3.5-0.8B 4x 0/162 (machine) |
| **Attention Q/K** | `C07.T5.S4.L12` attn_k 12/15; `C07.T1.S4.L12` 11/15; `C06.T1.S4.L12` attn_q 10/15 (Claude, LFM2) | T1 "stores 1.875x at M16 (storage trap)"; FP32 target-lattice Q/K 0/96 (`pg-jul-b:F13-qk-only-insufficient`) | three stacked attention folds collapse (REALIGNMENT §3c) | — (see attn_qkvo) | LFM2 only |
| **Attention V/O** | `C08.T8.S4.L02` attn_v 9/15; `C09.T3/T4/T5.S4.L02` attn_o 9/15 (Claude, LFM2, one site, about −3 cases) | — | `C09.T8.S4.L02` 5/15; `C08.T1/T2` 5/15; V/O "won't follow q/k down" | 21 (attn_o) + 10 (attn_v) | LFM2 only. Note: the technique row says T8 is "dead at attn_v/attn_o", but `C08.T8.S4.L02` reads 9/15; the two disagree |
| **Residual attention QKVO / V-O** | **none as a fold.** `p35-qkvo-L19G4`: held-out top-1 56/64 → 64/64, four-token rollouts 5/8 → 8/8, ~0.08% of parameters at one site, **Adam/backprop**, machine-scored, no checkpoint | head isolation TF top-1 11.67% → 65.75% with behaviour 0/1000; 1.41B guided QKVO first-action 50/192, exact 0/24 | LFM2 10% producer-first residual-attention MLP manual 0/7; V-only cache lost parent top-1 on every trace; Q-DEIM 4x all 37 fail | 112 of 193 (attn_qkvo) | Qwen3-4B (one site), LFM2 (none) |
| **Heads (merge) / head_dim** | `ARCH.KV6JQ` 16 → 12 q / 8 → 6 kv by joint relation solve: 8/15 at 0.979x (Claude, LFM2); `M5.headmerge.L02/L12` 6/15 (machine) | head merge is exact (identity 6.7e-16) but "8-13x more drift per parameter" (`P1.head-merge-expensive`); Qwen3.5-4B GDN head_dim 128 → 32 (`OC01-F-Q35-4B-G5D32`) 5/15, B 5 at 0.812x (**machine**); toy KDA heads/dims (`W11A.kda.*`) mechanics | `ARCH.KV6` averaged 5/15; HD16 0/15, HD48 1/15 (`MX43.F.band-axis-resealed`); head merge on a vocab base 2/15 (`MX27.F3`) | 39 (heads) + 17 (head_dim) | LFM2 (read); Qwen3.5-4B (machine); K2KV refused because of a q_norm −1.0 divide-by-zero, never reviewed |
| **Vocab row coreset** | `ARCH.V19` 14/15 at 0.795x (65,536 → 19,456 rows); `ARCH.V12` 11/15 at 0.763x (K = 12,288); `ARCH.V10` 10/15 (Claude, LFM2). Cyril ruled it counts as folding, "[Cyril quote withheld]" | ladder K 32,768 11 / 20,480 13 / 16,384 9 / 12,288 11 / 10,240 10 / 8,192 4: a cliff, not a slope (`gap2:w25-vocab-axis-cliff`) | `LFM2.vocab-0.48x` 2/15, which Cyril read as "a dead model" (ev 575550). LFM2 35% vocab: top-16 overlap .868 with 0/11 tasks | 105 of 180 | LFM2 only. Qwen C1 failed mechanically on mandatory token 248044 and stayed off; never behaviour-tested |
| **Cross-tokenizer bridges** | none | mechanical spelling bridge maps all 64,402 donor ids to 1-15 target spans (`pg-jul-b:F21`); K3 ↔ Qwen `boundary_map` (byte-boundary pairs) used for K3 KD and the C17 fits | v4096 bridge alone cuts parent-correct 8 → 3 (`SSL.F5`); "proven cross-tokenizer transfer" **refuted** (`f-xtok-4b-transfer-unproven`); no SmolLM2 pair bridge can exist (235/256 byte atoms) | — | arbitrary-pair bridging does not exist (`tokenizer-bridge-donor-subset-only`) |
| **Experts → dense** | **none** (0 reader-read arms in the DB) | K3 dense pass, all 92 MoE layers + L0, folded prefix, record mode: held top-1 0.692, top-5 set 0.596, NLL 2.157 vs teacher 1.261 (§9). C10 union in S3 R-Q 180-187/210 vs trunk; S5 "no measurable held-out text benefit" | C10 legacy full-weight target collapses the trunk (0/210 open-loop, 7/210 rehearsal); R-Q numbers only | 57 of 57, all unreviewed | K3 only |
| **Cross-family transplant** | none read by an agent | Qwen3-0.6B host + LFM2.5 MLP at alpha 0.005: "Passed" (machine) at **0.00% shrink**, i.e. almost nothing transplanted; with depth fold 5.28% (machine 0.75, "more repetitive than the host") | SmolLM2 L6 prefix "the honest headline ... negative" (16% top-1); Qwen3-4B → 88-134M blanks 0 behaviour (`cross-family-blank-zero`); 146M initializers all 0/1000; tiny-alpha transplant closed | 97 of 97, all unreviewed | Qwen3/LFM2/Pythia/SmolLM2 at toy alpha |
| **Kind translation** | none | GQA → CSA summary split, stride 4: 99.5% inside the operator class (toy, `W11A`); conv → attention lossy (0.77-0.83) | attention → ShortConv "not viable" (far mass 0.52-0.69) | 17 of 17 | no receipt for KDA → GDN, MLA → GQA or SiTU → SiLU (`k3-inspired-fold-tests-untested`) |
| **Modality units** | none as a fold | raw-image patch graft into a Qwen3-0.6B merged host: "useful generated text answers", synthetic task, gradient-fit; S7 units MLP-only 4x, whole blocks 0.50-0.69; C15 VAE absorption: every audio arm misses the line | S7 H3 output worse than its noisy input by 3.5-4.4 dB video and 9.5-14.3 dB audio; OMNI1 text 5/15 A, 0/15 B; QIMG B30 refold 0/60 targets (numeric) | 94 of 102 | media DiTs, mechanics only |

---

## 2. Depth: excision and merge

**Proven on behaviour (LFM2.5-230M, frozen-15, Claude + machine)**
- Single-block census at 0.947479x: block 13 13/15 (`ARCH.X13A`, above the parent's 12/15), block 7 11/15, block 9 10/15. Block 0 (`ARCH.X00`) is 0/15 at the same 12.1M parameters; attention blocks 2 and 12 are each 0/15 (`MX16.F3`, `MX/wave16/NOTES-MX-T.md:324-328`; `MX/wave17/NOTES-MX-U.md:56-58`). X13A also has lambada perplexity 4.4x worse (24.39 → 106.78).
- Ladder: 1 block 13/15 → 2 blocks 7/15 (`ARCH.P1301`, 0.895x) → 3 blocks 3/15 (`ARCH.T010713`, 0.842x) → 4 blocks 0/15 (`ARCH.Q01050913`, 0.790x) (`MX19.F7`).
- For pairs, cost is set by the distance between the blocks, not by their solo scores (`ST.pair-distance`). The law does not hold for three blocks (`PHASE1:432-440`).
- Block choice decides which capabilities survive: X09 and X13 share 0 of 16 lanes (`R3.L6-blocks-choose`).
- Locally re-solving after an excision does nothing (X05A update_scale 0.029, still 6/15), and a trajectory objective is harmful (X05T 0/15) (`MX15.F7`, `MX15.F3`).
- The census ranked blocks backwards for two waves (`MX16.F2`). Site choice has to be measured, not predicted.

**Machine upper bound (Qwen3.5-4B, OC01, 2026-09)**
- Results by layer kind:
  - full-attention X03 and X15: 15/15 (B 13);
  - GDN X08 and X16: 15/15 (B 12 and 13);
  - full-attention X31: 12/15 (B 10);
  - GDN X00: 10/15 (B 10).
- 0.950-0.951x on the header count (`REV/live/SCORECARD.md:27-33`). The 120,599,552-parameter accounting gap (MTP) is still unexplained. "Nobody has read this child" (`oc01/human-review/.../UPPER-BOUND-NOTICE-OC01-F-Q35-4B-X03.md:5-8`). The children were deleted by about 10-21; the review packages remain.
- **Scaled from LFM2 → Qwen3.5:** one layer can go at most interior sites, and the first layer is the costliest on both (LFM2 X00 0/15; Qwen X00 10/15, the lowest of the ladder). The ladder was never extended past one layer on Qwen.

**Merges**
- The only gated merges are machine choice gates on Qwen3-0.6B: 13+14 at 2.64% and width 0.995 + fold 13+14 at 2.848%; and 13+14 + 15+16 at 5.28% on the LFM2.5-transplant host. Later groups and multi-group merges failed the gate (`VALIDATION_RESULTS_2026-06-26.md:405-410`). Those gate failures are numeric, so they count as unreviewed.
- Closed path: every depth-sharing scheme collapsed; "averaging parameters is not ordered function composition" (`ra-law-depth-sharing-closed`).
- The v2 proxy period merge (4:1 from the retired T line, about 137x) was read by Claude as 0/9 in each of three arms.

**What v2 needs and the record does not have:** K3 92 MoE layers → 24 (about 3.8:1 by the 92:24 floor-interval pooling in `V2_FOLD_INIT_PLAN_2026-10-21.md:110`), Q38 64 → 24, or for the 4B 92:32 / 64:32. The best evidence is one layer of 32 on Qwen3.5-4B (machine) and two blocks of 16 on LFM2 (7/15). No excision ladder exists on Q38-27BU, K3 or any media parent (record §3d).

---

## 3. Hidden width

- **Proven (LFM2, Claude):**
  - W896 4/15 at 0.859x and W768 3/15 at 0.723x. W736 reads 1/15, and W720 and everything below read 0/15 (`ARCH.W512`, `P1.cliff-slope`).
  - The cliff is a 16-dimension slope, 736 → 720. Wave 21 withdrew the earlier "the cliff is a step" (record §4d; ARCH_DRIFT's "below roughly 0.86x" is wrong, record §10 item 8).
- **Basis:**
  - The pooled reader moment's top eigenvectors are the only basis that produced a model; five, then six, alternative objectives lost to it (`ST.subspace-point`).
  - The currency is read energy, not stream energy (`P1.read-currency`). Below about 0.90 of read energy every child is dead; above it the fraction does not predict (0.9178 → 0/15, 0.9193 → 3/15).
  - There is no low-rank residual structure: the top 768/1024 directions hold 0.9389 of read energy (`MX19.F1`).
- **Composition:** width with anything else is dead. `W896KV6JS` 2/15, `W896X13` 1/15, and every W896+CL1792 child 0-1/15 (`MX19.F6`, `w36.width-family-kills`, `gap2:realign-interference` "WIDTH x MLP REGION CLOSED").
- **Scaling:** none. Qwen3.5-4B has one hidden-width arm and it is not alive. The T-line arms (`V2FOLD.proxy`) are dead and retired. SmolLM2 576 → 432 covered two boundaries only.
- **Gap to v2:** v2 needs 7168 → 2048 (0.286) from K3 or 5120 → 2048 (0.40) from Q38; the 4B needs 0.357 / 0.50. The lowest live hidden-width child on record is at 0.75 of width, on a 230M model, alone. The planned width fold (one orthonormal stream chart per source, `V2_FOLD_INIT_PLAN:85-93`) has only one behaviour result, the dead 137x proxy.

---

## 4. MLP width: per-site coreset and its scaling

- **Proven per site (LFM2, Claude):**
  - `C13.T8.S4.L05` (mlp_down, Gram relation matching) 12/15 at component 0.2498. `C11.T1.S4.L00` 10/15. `COMP.MA.MXJ1` (that one C13 component installed) 12/15 at 0.991x stored.
  - Global intermediate `CL1792` 7/15 and `CL1792M` (reader-metric selection) 8/15 at 0.856x, against W896's 4/15 at matched ratio (`FR3.MLP-beats-width`). CL1792M robust-re-reads 8 → 0 (`MX33.F1`).
  - "Per-site MLP channel coreset at 4x" in REALIGNMENT mislabels the cells: only C13 is MLP, and none is the C2 channel coreset (record §10 item 4).
- **All-site fails, with a cliff:** 14 SwiGLU coresets, each with held write cosine 0.81-0.94, compose to 0/15 (`MX14.F3`). At 2x per site (ff1280) the child was "degraded, not degenerate, ~5/15". Global intermediate ≤640 is dead (`MX.ffn-cliff`).
- **Scaling of C2 MLP 4x on all layers (no training):**

| Parent | Child | Score | Reader | Source |
|---|---|---|---|---|
| LFM2.5-230M | ff640 (4x, 14 sites) | 0/15 | Claude | `MX14.F3` |
| Qwen3.5-0.8B | `x3-08b.c2k896` | 0/162 content vs parent 54/162; 158 artifact misses (empty/capped/looping) | machine | `DENSITY_AND_DATA_PROGRAM_2026-10-14.md:273` |
| Qwen3.5-4B | `OC01-F-Q35-4B-C2K2304` (0.610x) | 5/15, B 2 | machine | `REV/live/SCORECARD.md:34` |
| Qwen3.5-4B | `C2K1152` (8x, 0.549x) | 1/15, B 0 | machine | same |
| Q38-27BU | `OC02-F-Q38-27BU-C2K4352` (0.523x, MTP removed) | 8/15 sealed, B 1, against parent 15/15 | Codex sealed + machine | `CHIMERA_OMNI_FULL_FIRST_FOLD_STATUS_2026-10-06.md:582` |

  Retention rises with parent size (0 → 5 → 8 of 15), but column B (copy-backs refused) stays at 0-2 everywhere. Nobody has read any of these children for intelligence; "machine upper bound" applies to all of them. With LoRA KD + OPD, the 0.8B child recovered to 14/162 (`x3-08b.kd1m-opd-*`), and LoRA is ruled out for Omni (D 121).
- **Media MLP:** H3 L25 MLP 4x: component holdout relL2 0.3034 vs prior 0.3266; whole block 1.81x (relL2 only). S7 "4x" is MLP-only (`gap2:s7-media-4x-is-mlp-only`).

---

## 5. Attention: Q/K/V/O, residual attention, heads

- **Per component at 4x (LFM2, Claude, one site each, parent 12/15):** attn_k `C07.T5.S4.L12` 12/15 and `C07.T1` 11/15; attn_q `C06.T1.S4.L12` 10/15; attn_v `C08.T8.S4.L02` 9/15; attn_o `C09.T3/T4/T5.S4.L02` 9/15. With Q/K one site costs 0-2 cases, with V/O about 3. Three stacked attention folds collapse.
- **Head merge:** exact but expensive. `ARCH.KV6JQ` (joint relation solve, value path off) 8/15 against averaged `KV6` 5/15. Choosing the value-path lambda out of sample took a merge from 3/15 to 7/15 (`MX18.F2`). One merged attention block costs more than two whole SwiGLU lanes plus the token interface (`MXQ.F4`). On a vocab base the merge reads 2/15 (`MX27.F3`).
- **head_dim:** LFM2 frequency-band HD16 0/15 and HD48 1/15, so the axis is re-sealed. Qwen3.5-4B GDN head_dim 128 → 32 (`G5D32`) 5/15 B 5, machine. The `K2KV` (Qwen3.5-4B KV merge) zero gain came from a q_norm coordinate whose bf16 weight is exactly −1.0 (L27), giving a divide-by-zero. It was never reviewed and is unreviewed, not failed.
- **Residual-attention folding:** see the table in §1 and record §5. The only positive is `p35-qkvo-L19G4`: 5/8 → 8/8 exact four-token rollouts, gradient-fit, about 0.08% of parameters, one site. First-action gains never became trajectory coherence (`ra-law-aggressive-child-first-action-not-trajectory`). No Oct Omni fold used residual attention. **Nothing on record supports calling residual-attention folding "proven" as a fold technique.**
- **v2 mixer plan vs record:** `V2_FOLD_INIT_PLAN` selects whole heads by write energy and ridge-refits `out`. Selection with an exact per-head move is mechanics. Ridge-refitting `out` against pooled writes is the T1/T2 family, proven only one site at a time on LFM2. Q38's GQA (24 q / 4 kv × 256 per the plan, unverified at its P0) → v2's 8 / 2 × 256, which drops 2/3 of q-heads and half of kv-heads per layer. The nearest proven step is LFM2 16 → 12 q heads (8/15).

---

## 6. Vocab coreset and cross-tokenizer bridges

- **Proven (LFM2, Claude):** `ARCH.V19` 14/15 at 0.795x, the best review in the campaign, though Cyril asked for the same "[Cyril quote withheld]". The method is a zero-solve verbatim row coreset: every special token + 256 ByteLevel bytes + BPE merge closure, remapped to a stock tokenizer. The ladder has a cliff inside 2,048 rows (K 8,192 4/15 against 10,240 10/15).
- **Composes once:** `ARCH.V20X13` 10/15 at 0.747x, and `V20X13G03` (+ terminal gain) 11/15 at 0.7466x. The sibling `V19X13G03` robust-re-reads 11 → 2. Measured interference: depth × vocab 4.44 cases, the largest pair.
- **v2 relevance is small.** v2 keeps the 248,320 Qwen text vocab + 512 control rows tied. Cyril: vocab "[Cyril quote withheld]". The Qwen C1 vocab coreset failed mechanically on the mandatory token 248044 and was never behaviour-tested.
- **Bridges:**
  - K3 enters through `boundary_map.json` byte-boundary pairs on a 24 × 4,096 corpus. That corpus was used for both fold fits and C17 charts, so it is contaminated for any held claim on S5/OMNI1 (`gap2:misc-fold-evidence-contaminated`).
  - "Proven cross-tokenizer transfer of 4B-class donor intelligence" is **refuted**.
  - A bridge by itself costs retention (`SSL.F5`: 8 → 3).

---

## 7. Experts → dense (K3)

See §9. Every experts → dense arm in the DB (57 dropped, 47 works/partial) has **no reader read**. Every number is a teacher-forced agreement, KL, NLL or relL2. The C14 expert-union coreset is the same coreset machinery as the MLP C2 path, applied to 896 routed + 2 shared experts.

---

## 8. Cross-family transplant and modality units

- **Cross-family:** the "Passed" transplants are tiny-alpha (0.0005-0.005) LFM2.5/Pythia MLPs added to a Qwen3-0.6B host at 0.00% shrink, judged by a machine choice/PPL gate. They show the host survives a small addition, not that intelligence moved. Real cross-family prefix folds (SmolLM2, 08-03/04) are negative past layer 3: "the wall is pretraining" (`pg-aug-a:F-crossfamily-wall-pretraining`). There is no reader-read cross-family arm.
- **Kind translation:** toy GQA → CSA works at stride 4; conv → attention on real LFM2 is lossy; attention → ShortConv is not viable. The kinds v2 needs (K3 KDA → GDN, gated MLA → gated GQA, SiTU → SiLU) have **no receipt** (`k3-inspired-fold-tests-untested`, open). The v2 plan's F-K3MIX/F-K3MLA arms (collapse per-channel decay to `mean(A_log)`; compose `kv_down·kv_up`, zero-pad 128/192 → 256, NoPE → partial RoPE) are design only.
- **Modality units:**
  - **Proven, but trained:** from-scratch AV flow toys (`pg-aug-b:omni-avflow-v3-x0`, v4 evidence), and the June raw-image patch graft (gradient fit, synthetic task: "does not prove general vision ability").
  - **Folded media from parents:** S7 units are MLP-only 4x with whole blocks at 0.50-0.69, and their outputs are worse than the noisy input. C15 absorption missed its line on every audio arm, and H3 video meets the line only because the parent's own round-trip is 9.91 dB. ACE L12 holdouts leaked 28% by batch row.
  - **Not on record:** a raw-modality input boundary or x0 output head folded from any parent. The v2 plan's `c22_boundary_head_fold.py` is new code.

---

## 9. K3: the record, re-read from the live receipt

**What exists** (read-only probe 2026-10-29 of `/home/cereal/omni-campaign/k3-run-oct09-claude`):
- `fold/dense/L00..L92`: 93 directories.
- `fold/resid`: 32 directories, L61-92 only.
- `receipts/k3_dense_pass_receipt.json`: 3,266,608 B, dated Oct 10.
- Tool `k3_dense_pass.py` 1.2.5, run `04db523ceafa`.
- The NAS archive `k3-run-oct09-claude-archive` exists.
- Protected by `NAS_CLEANUP_2026-10-27.md:181`.

**What it is**
- All 896 routed + 2 shared experts of every MoE layer, as a C14 expert-union coreset (form E, K = 17,408 latent channels), 29,784,936,832 → 370,675,200 parameters per MoE layer (80.353x).
- L0 is the dense MLP, verbatim.
- Every layer was fitted sequentially on the folded prefix's rows against the teacher's routed output.

This is the structure of Cyril's 09-02 ruling ("[Cyril quote withheld]"), with one exception: the chart-merge across the Chimera trunk was never done for v2.

**Recorded verdicts (record mode: every layer installed)**
- 24 MoE layers carry no failing gate: 1, 2, 3, 4, 5, 7, 8, 12, 13, 15, 24, 25, 26, 27, 48, 49, 52, 60, 61, 82, 83, 84, 86, 87. L0 is verbatim.
- 68 are "refused":
  - G3 alone (folded prefix no better than a zero child by the 0.95 line): 37
  - G1 routed write + G3: 27
  - G1 alone: 3
  - G2 held/fit > 1.5: 1 (L06)
- These counts come from the live receipt's `layers.*.acceptance.failing` and match the record's 24/68.
- **Under Cyril's 10-06 rule these are recorded numeric verdicts, not failures.** Every one of the 93 layers is installed, and none has been judged on behaviour.

**End-of-stack numbers (teacher-forced only)**

| Split | Positions | top-1 agreement | top-5 set agreement | KL mean | NLL child / teacher |
|---|--:|--:|--:|--:|--:|
| fit | 69,725 | 0.755 | 0.667 | 0.504 | 1.758 / 1.373 |
| held | 23,180 | 0.692 | 0.596 | 1.000 | 2.157 / 1.261 |
| review rows | 47+ tokens per row | 0.304-0.421 | 0.391 (row 0) | 2.12 (row 0) | ~3.9-4.4 / ~1.7-1.9 |

The receipt's own note: "nothing here is a free-run decode". **Never decoded; no behaviour review** (`pg-oct:F-k3-never-decoded`, `k3-folds:k3-fold-never-decoded`, open).

**Local evidence**
- L62-80 routed writes sit near the zero map (held relL2 0.76-0.92 against 1.0), while the latent fit is 0.35-0.64 (`k3-folds:late-stack-routed-write-near-zero-map`).
- L1-4 recover about 59% of y_lat energy and leave a folded prefix at relL2 0.34-0.46 (`pg-sep:F-K3-L01-L04-thin`).
- The receipts cannot separate dense capacity at K = 17,408 from prefix drift (`k3-folds:dense-capacity-vs-prefix-drift-unseparated`).
- The c14 veto structurally refused healthy dense replacements, so lambda moved to argmin check-SSE (`pg-sep:F-c14-law-wrong-scale`).
- The teacher-cache tool is identity-exact on 93/93 layers, while B2 bf16 parity fails (p99 1.25, top-1 0.9635 < 0.97). Reading that as a bf16 accumulation diagnostic is supported, not proven.

**K3 into a trunk (27B line, retired)**
- C10 a003 (w = 0.03) gives no measurable held-out text benefit (R-Q 110/210 vs parent, S3 108/210). The full-weight union collapses (0/210 open-loop, 7/210 closed-loop).
- The charted K3 term is about 18.5x the trunk MLP output.
- The coreset kept almost only routed channels.
- All of this is token agreement, with no decoded text.

**Lineage:** L1-4 gate 09-07; L5-35 Codex 10-06; L36-60 Codex 10-07; L61-92 Claude 10-09/10. The receipt's `prefix_producer_acceptance` lists four accepted producer tool sha256s, so the prefix was produced by several tool builds.

### 9a. What a K3 all-experts, all-layers fold with folded prefix would need for v2

From the record and the v2 plan, in order:
1. **Decode the existing K3 dense child before anything else.** `k3_review_stream.py --mode decode --dense-dir ... --rung 17408` exists ("greedy decode once every MoE layer is folded", `OCL/k3/README.md:16, 82-92`) and has never been run on `fold/dense`. Run it on the frozen 15 + bank 2 with a K3 parent baseline stamped by `--emit-baseline`, and have a reader judge the outputs. Until that read exists, "K3 primary" in any fold is unproven. On resources (estimate, not measured): the dense FFNs are about 76 GB and the non-expert body comes from `/mnt/lexar/models/moonshotai/Kimi-K3` streamed per layer. Host MemAvailable read 193.8 GB at 02:04Z, so a CPU decode with the ≥ 40 GiB floor looks feasible but is unmeasured. GPU6 only through the gates.
2. **Treat the 68 refusals as unreviewed.** If the decode is alive, the per-layer gates are diagnostics. If it is dead, the record cannot say whether capacity (K = 17,408) or prefix drift is the cause. A wider rung (the 32,768 or 65,536 rungs the K3 README lists) on the late stack (L62-80) is the separating test.
3. **Refold the K3 rows L1-60.** Only L61-92 rows survive. `k3_dense_rows.py` (new) is about 70 min on one P40 per `V2_FOLD_INIT_PLAN:252`, but the P40s are held by run 0, so this needs another lane or CPU.
4. **Chart K3 onto v2, not onto T.** The C17 K3 → T charts were deleted 10-21, and T is retired. v2 needs K3 7168 → 2048 directly (the plan's arm F-K3DIRECT, about 12 h CPU serialised into the loop) or through a new parent-derived trunk chart. A 0.286-of-width chart is far past anything alive on record (§3).
5. **Pool depth 92:24 (or 92:32).** Three or four K3 layers go into each v2 layer's FFN union coreset (6,144 kept of ≥ 52k K3 channels per pool). No depth pooling at this ratio has a behaviour result. The nearest is the dead 4:1 proxy.
6. **Set the K3 weight from behaviour, not from the 27B sweep.** `w_k3 = 0.03` was chosen on R-Q token agreement in the retired S3 trunk, where K3 added nothing measurable. v2 needs its own sweep, judged on decoded outputs.
7. **Mixers:** K3 KDA (96 heads × 128, `A_log` F32[128] vs 96 heads unsettled, D-a2b) and gated MLA (NoPE) have no fold receipts into GDN / gated GQA. Either fold the attention from another parent (Q38-27BU) or build F-K3MIX/F-K3MLA as tests.
8. **Objective:** add the top-n (10+) set target at block ends (D 821; the X2 design). The fit is currently relL2 on y_lat plus prefix relL2 only.
9. **Fresh held data:** the 24 × 4,096 dual-tokenized corpus is contaminated for held claims. A new held split with K3 pairing is needed.
10. **Export:** no K3-derived child was ever exported. The MoE half of the export spec is unimplemented, and a GGUF needs AttnRes and finite SiTU betas (`f-no-k3-child-to-export`).

---

## 10. Composition and scaling summary

**Composition (all LFM2)**
- First step survivable, second fatal (`P1.additive-interference`): V16X13 13/15 + 9/15 on disjoint tensors → 5/15.
- Survived once: vocab + excision 10/15 (`ARCH.V20X13`); + terminal gain 11/15 (`V20X13G03`); two components at 8x + 4x 13/15 (`MXK14`, parent-shaped).
- Fatal: width + head merge 2/15; width + excision 1/15; W896+CL1792 families 0-1/15; every three-block child 0/15 except T010713 3/15; head merge on a vocab base 2/15. A second fold on a shelf base costs excision −3, head merge −11 and width −11 (`MX27.F3`).
- Each independently solved body component costs about 3 review cases (`ST.five-3`). Local fit does not predict composed behaviour (`ST.five-4`, `S1C.the-law`). Seven components at isolation free-run 0.738-0.994 compose to 0.539.
- Whole-body exact-4x contraction is 0/15, and so is the 2x arm (body cliff in (1.0x, 1.559x], `F-body-contraction-cliff`). Cyril overruled the "stage 1 complete" claim (ev 67d243e0...).
- REALIGNMENT's "one structural axis survivable; two fatal" is mis-stated (record §10 item 3). Vocab + excision survives; width plus any second axis does not; three structural folds do not.

**Scaling across parent families**

| Technique | LFM2.5-230M | Qwen3 / Qwen3.5 (0.6B-4B) | Q38-27BU | K3 |
|---|---|---|---|---|
| One-layer excision | 13/15 at site 13; 0/15 at sites 0/2/12 (Claude) | Qwen3.5-4B 10-15/15 (machine) | none | none |
| Layer merge | — | Qwen3-0.6B 2.64-5.28% shrink (machine gate) | proxy from T dead (Claude 0/9) | none |
| MLP coreset 4x all layers | 0/15 (Claude) | 0.8B 0/162 (machine); 4B 5/15 (machine) | 8/15, B 1 (Codex) | C14 union all layers, never decoded |
| Hidden width | alive to 0.75 width (Claude) | one arm, not alive | proxy only (dead) | none |
| Heads / head_dim | merge 8/15 (Claude) | GDN head_dim 4x 5/15 (machine); K2KV harness bug | none | none |
| Vocab coreset | 14/15 at 0.795x (Claude) | C1 mechanically failed | none | — |
| Residual-attention QKVO | none | Qwen3-4B one site, trained | none | none |

What scaled: single-layer excision (with the first layer still the costliest) and the MLP coreset's retention rising with parent size (0 → 5 → 8 of 15 at 4x MLP). Nothing scaled to K3 on behaviour because nothing on K3 has been read.

---

## 11. Dropped on numbers = unreviewed (Cyril's rule)

Query (`omni_lab_ro`): arms with `verdict_class in ('fails','closed_path')` and no `fold_behaviour` row with a score from claude, codex, cyril or mixed. Machine gates (choice/PPL/relL2/acceptance) are counted as numbers.
- **All programs:** 2,217 of 2,726 dropped arms have no reader read. 1,091 have no score at all.
  - By program: early-surgery 331/331, cross-width 206/206, cross-family 206/210, media 207/213, lfm2-dynamic 193/248, chimera-omni 124/130, k3 78/78, cpcf 26/26, density 35/35, matrix-campaign 299/507, neural-forge 267/439, residual-attention 130/163.
- **By axis** (component kind; dropped unread / dropped):

| Axis | Unread / dropped |
|---|---|
| mlp_width | 410/627 |
| whole_model | 567/625 |
| hidden_width | 117/278 |
| attn_qkvo | 112/193 |
| vocab | 105/180 |
| modality_unit | 94/102 |
| cross_family | 97/97 |
| depth_merge | 90/197 |
| experts_to_dense | 57/57 |
| heads | 39/69 |
| depth_excision | 38/103 |
| head_dim | 17/22 |
| kind_translation | 17/17 |
| cross_width | 19/19 |

- **Works/partial arms that a reader actually read:**

| Axis | Read / works-partial |
|---|---|
| depth_excision | 27/64 |
| depth_merge | 26/53, all mixed and after recovery training |
| hidden_width | 28/75 |
| mlp_width | 77/283 |
| attn_qkvo | 26/56 |
| vocab | 34/100 |
| heads | 13/35 |
| experts_to_dense | 0/47 |
| cross_family | 0/22 |
| kind_translation | 0/16 |
| head_dim | 0/7 |
| modality_unit | 8/45 |

- Specific numeric drops that matter for v2:
  - the K3 68 "refused" layers (installed anyway);
  - the Qwen3-0.6B merges that failed only the choice gate (19+20, 8-9+19-20);
  - Qwen3-0.6B x0.995 FFN (PPL ratio 1.595 > 1.25);
  - QIMG B30 (0/60 targets met);
  - the OC01 refused variants G5V16K8, N48, K2KV and K2FG (never reviewed; K2KV was a harness divide-by-zero);
  - the census-killed matrix cells ("KILLED BY CENSUS/PREDICTION without a built child");
  - C15 PSNR-line misses.

  None of these is a behaviour verdict.

---

## 12. What a v2 fold pass can take from this

**Use as proven, within scope**
- **Excision:** one layer at a measured site, not a predicted one. Expect the first layer to be costly.
- **Per-site 4x component folds:** T8 on MLP down and conv, T5 and T1 on attn_k/q. Install one or two, not all.
- **Bases and selection:** the reader-moment basis for any width chart; out-of-sample lambda selection (`MX18.F2`); prior-anchored ridge with measured, not law-based, acceptance.
- **K3 dense pass:** the machinery (sequential folded-prefix fits, streamed experts, record mode) works mechanically on all 92 MoE layers.

**Must be tested before they can carry a 2B/4B trunk**
- hidden width at 0.29-0.50;
- depth at 2.7-3.8:1;
- MLP coreset at every layer composed with width and depth;
- head reduction 24 → 8 q (Q38) composed with width;
- K3 decode and a reader's judgement;
- K3 → v2 charts;
- KDA/MLA kind translation;
- raw-media boundary and x0 head folds;
- top-n as a fit target;
- any child with three or more structural axes.

On the record, the planned fold (`V2_FOLD_INIT_PLAN` structure, built from parents) is a new experiment, not an application of proven techniques. Run 0 is its comparison.

**Open questions for Cyril (none answered in the record)**
- Whether the folded trunk must pass a review on its own (D 41, D 106) or is judged after distillation from the parents (D 116).
- The copy-back and PROBE rulings (`cyril-scores:F08`).

---

## Verification and limits

- **DB queries run (read-only, `omni_lab_ro`):**
  - per-arm scorer/score strings for about 55 named arms;
  - `v_technique_scoreboard` aggregated by kind;
  - K3, cross-family, bridge, kind-translation, head and composition findings with full statements;
  - the dropped-unread counts by program and by component kind (§11);
  - axis coverage by parent name.
- **Live check:** `ssh_client.py --server ai` read-only. Directory counts (dense 93, resid 32), the receipt's per-layer `acceptance.failing` tally (none 25 = L0 + 24; G3 37; G1+G3 27; G1 3; G2 1), `end_of_stack` fit/held/review values, and MemAvailable 193,769,440 kB.
- **Not verified here:**
  - the record's source-line citations (taken as the record states them; C01-C13 were verified on 10-28, later claims were not);
  - whether the `C08`/`C09` attention cell mapping matches the technique-row claim that T8 is "dead at attn_v/attn_o";
  - the CPU feasibility of a K3 dense decode (estimate only);
  - the parent counts behind the §11 per-axis numbers. Arms carry several component kinds, so axis counts overlap, and kinds come from free-text matching.
- **Deliberately left out:** training-only results except where they bound a fold (FULLKD run 1, X3 LoRA recovery); individual matrix cells beyond the best per component; media C16/C17 numeric detail beyond §8.
