# CHIMERA OMNI CAMPAIGN — RUN BOOK

Campaign root (repo mirror): `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/` (= `OC/` below)
Campaign root (server, local disk): `/home/cereal/omni-campaign/` (= `$OC_ROOT`)
Written: 2026-09-02, from the Phase-0 reader digests (R1–R6), the six lane deliverables and their two adversarial review rounds, the chimera-import lane status (S1–S5), `PHASE0_RESULTS_2026-09-01.md`, and `CHECKPOINT_POLICY.md`.
Plan of record: `docs/CHIMERA_OMNI_FOLD_PLAN_2026-09-01.md` (sections 3b, 5, 11, 12). Protocol: `OC/R3-review-ledger-protocol/REVIEW_LEDGER_PROTOCOL_SPEC.md`.

Every number in this book is tagged where it matters: **[verified]** = read out of a shipped receipt/log or measured on the box this session; **[assumed]** = inference not yet receipted; **[proposed]** = a decision line for Cyril, never a gate anyone invents.

---

## 0. Standing rules that bind every command in this book

1. Server access ONLY via `python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script '<bash>' --timeout <=280` (or `--upload`/`--download`). One process holds the SSH lease; on `[SSH BUSY]` sleep 20 s and retry, up to 10 times. Long waits are one server-side `until ...; do sleep 10; done` inside one call, never a client-side retry loop (the calibration lane lost hours to orphaned loops).
2. Every remote command wrapped in `timeout N`. Long jobs launched detached with `setsid nohup ... < /dev/null &` or `systemd-run --user --unit=oc-<lane>-<tag>` and polled by reading their log/receipt.
3. Never recursively walk `/mnt/ai_data` (CIFS, 5–25 MB/s). Stage what you need to `/home/cereal` (537 GB free at last `df`; 1.42 TiB K3 experts cannot be staged — stream them).
4. GPU pinning by PCI bus id and UUID (`CUDA_DEVICE_ORDER=PCI_BUS_ID`); torch index 6 without that env is a P40. UUIDs used by the lanes: GPU0 3090 `GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561`; GPU4 3090 `GPU-2ce6a95c-7b1f-644a-3808-d781732a0582`; GPU6 RTX 8000 `GPU-f5a09535-fc85-a027-50f7-734b344747dd`; P40 GPU2 `GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4`; P40 GPU3 `GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8`; P40 GPU5 `GPU-ce4a1da0-...` (export lane parity).
5. Never stop/kill a process or container you did not start; never delete on the server except through `oc_checkpoint.py delete` (R4). GPU4 and GPU6 only after the relay claim is confirmed (section 2.0).
6. Folding rules: closed-form only, >=4x per component, no quantization in a fold, storage counted in parameters, artifact-tolerant grading (loops/echo/cap-rides/missing-EOS never count against intelligence), every threshold a proposed decision line, verify artifacts by hash never exit code, dense only, every fold class stays OPEN until a built child closes it.
7. No ORM/SQLAlchemy/Pydantic anywhere in campaign code (the lab venv carries pydantic 2.13.5 as a vendor dep of compressed-tensors — never import it). No hardcoded secrets; vault keys only.
8. Writes only under `OC/<lane>/`, the scratchpad, or `/home/cereal/omni-campaign/<lane>/`. Existing neural-forge and chimera-omni sources are never edited (the import lane lives on branch `claude/omni-import`; `main` untouched at `c1ca0b2`).

Common paths: `PY=/home/cereal/model-surgery-lab/venv/bin/python` (torch 2.11.0+cu128, transformers 5.12.1, fla 0.5.2, compressed-tensors 0.18.0, gguf 0.19.0, safetensors 0.8.0; `causal_conv1d`, `flash_attn`, `peft`, `bitsandbytes`, `deepspeed` ABSENT). Parents on local disk: `/home/cereal/models/{Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.8-27B-Uncensored}`; GGUF `/home/cereal/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`; K3 on the share `/mnt/ai_data/models/moonshotai/Kimi-K3` (96/96 shards, layer-0 shard staged at `/home/cereal/omni-campaign/import-lane/s4/k3_layer0_staged`); media models `/home/cereal/comfyui/models/{vae,diffusion_models,checkpoints,text_encoders,loras}`; ComfyUI core only inside image `comfyui:codex-minimax-h3-runtime-env-r5-20260808` (`/opt/ComfyUI`, `/opt/environments/python/comfyui/bin/python`). Curated dataset: `/mnt/ai_data/omnicoder_2026/datasets/MASTER_SHARD_INDEX_POINTER.json` — 631 entries / 94,318 rows / 640 shards at 2026-08-29 [verified]; the plan's "83,095 rows" is the stale June snapshot.

---

## 1. State of every lane

### 1.0 One-screen table

| Lane | Dir (OC/ … and /home/cereal/omni-campaign/ …) | Tests | Review status | READY for | Biggest gap |
|---|---|---|---|---|---|
| qwen-fold (Wave A harness) | `qwen-fold/` | 16/16 (server, CPU, random-init fixture) | round 2: **fix** (1 blocker: per-head Gram re-formation in `_refit_beta`/`_refit_decay`) | C1/C2/C3-4/C5/C6/C7/C9 builds + stock re-load + receipts | NEVER run on a real parent or a GPU; no C8 width; no 27B StreamedHarness; GGUF hook dry-run |
| ledger (registry/clone/delete/packages/dashboards) | `ledger/` + `CHECKPOINT_POLICY.md` | 95/95 (workstation), 95 w/ 1 skip (server) | round 2: **fix** (1 blocker: `..`/symlink path normalisation in protected-prefix guard + duplicate-directory check) | register/review/package/clone/delete/dashboards; verified against the 1,056-row ancestor ledger | ARCH/behaviour sealers not ported (ledger carries checkpoint rows only); bands.json absent |
| k3 (streamed harness + C14 coreset) | `k3/` | 29/29 (server, synthetic K3-shaped checkpoint) | round 2: **fix** (1 blocker: `cuda` vs `cuda:0` device alias zeroes the resident-memory estimate) | streamed per-layer pass, exactness receipt, C14 two-stage coreset, R-proxy/decode review | NO real K3 layer run; A_log axis (F32[128] vs num_heads 96) UNRESOLVED (D-a2b) |
| media (C15 VAE absorption, C16 ACE DiT transplant, H3-0) | `media/` | 56/56 (server) | round 2: **fix** (1 blocker: C16 coreset holdout must group by clip, not by 256-row batch) | real-VAE pairs for ACE/H3-audio/H3-video, ladder, composition into Chimera bridge/head shapes, ACE block-12 coreset 4.00x, H3-0 arms | ACE under-fit unexplained (13.3 dB below PCA bound); H3-0 separation = one gate, one seed; no decoded audio from C16; Hunyuan3D/Qwen-Image untouched |
| export (GGUF, parity, 256k envelope) | `export/` | 145/145 (both hosts) | round 2: **pass** | HF→GGUF→Q4_K_M/IQ4_XS, parity vs HF, envelope sweeps, KV-fit bisect, P40 sweep | imatrix not plumbed; P40 27B sweep still in flight (pre-fix module); `--runner docker` unexercised |
| calibration (parent bank runs, bank v2 draft, assessments) | `calibration/` | 77 (76+1 skip server, 71+6 skip workstation) | round 2: **pass** | frozen 15-case bank + safety, bank v2 DRAFT (30 text incl. 9 long-context, 5 vision, 5 AV spec), assessments, Q35-4B anchor with full v2 validity tuple | NOTHING GRADED by a human yet; vision/AV lanes never decoded; `--thinking off` surface still a v1 dossier; 200k does not fit bf16 HF on 24 GB |
| chimera-import (S1–S5, branch `claude/omni-import` HEAD `f86fe45`) | `chimera-import/` (reports) + repo branch | 376 (server), 201+78 skip (workstation) | S1–S5 each reviewed to **pass** | exact Qwen3.5 → Chimera import (fp32 max abs logit diff 3.19e-5, argmax 210/210), K3 layer-0 exact (3.47e-8), safetensors IO, qwen35 GGUF served by llama.cpp byte-identical 3/3 | MLA (layer 3) never measured; MoE tensors loader never executed; 4B/27B receipts not run; spec Q1 (`output_gate_type='swish'`) unresolved |
| R3 protocol spec, R1 datasets, R2 port spec, R4 import spec, R5 training constraints, R6 K3/H3 specs | `R3-review-ledger-protocol/`, `phase0-reader-digests/`, `qwen-fold/PORT_SPEC…`, `r4-chimera-import-lane/`, `reader-r6-harness-specs/` | — | specs | reference | — |

Round-2 blockers marked **fix** above are small, local edits (each < 1 hour); they are scheduled first in section 2 and must land before the affected tool is used on a real parent.

### 1.1 qwen-fold — the Wave A harness (`OC/qwen-fold/`, `/home/cereal/omni-campaign/qwen-fold/`)

Modules: `qf_arch, qf_corpus, qf_harness, qf_solve, qf_vocab, qf_mlp, qf_excise, qf_gdn, qf_attn, qf_termgain, qf_export, qf_build, qf_selftest, qf_evidence`, `env.sh`, `bodyq.sh`, `README.md`, `PORT_SPEC_lfm2_to_qwen3_5_2026-09-01.md`. Every file byte-identical local vs server (19 compared, 0 mismatches) [verified]. Pre-fix backup: `/home/cereal/omni-campaign/qwen-fold/.bak-20260902-011040/`.

READY: seven fold classes as single transactions producing a standard `qwen3_5` HF checkpoint plus a self-sealing receipt (`qf-build-receipt.json`, schema `neural-forge-omni-qwen3_5-standard-shape-child-v1`): C1 vocab coreset, C2 MLP channel coreset (verbatim gate/up rows, ridge-refit down_proj), C3/C4 excision with the above-diagonal causal-mask assertion (negative control fires at 0.4878 leak [verified]), C5 GDN value-head fold (count arm + dim arm; beta now refit through the logit, negative control reproduces the 3.70x Jensen understatement [verified]), C6 NoPE-dim contraction (rotary width provably unchanged; per-head conditional optimum gap 1.1e-16 [verified]), C7 kv-head merge (kv-only arm gate bit-exact; full-group arm), C9 terminal gain (identity gate exact). Solvers: row and Gram forms bit-identical (max_abs_difference 0.0 at N=8192,k=4352; 3.42x faster) [verified]; pivoted coreset argmax form 49x faster, 27B rung 17408→4352 completes in 97 s [verified].

GAPS: (1) never touched a real checkpoint or CUDA — every fold-quality number is on a random 8-layer fixture; (2) round-2 blocker: `_refit_beta` (qf_gdn.py:458) and `_refit_decay` (:385) re-form `X^T X` per value head — hoist to one Gram per layer via `prior_anchored_gram`; (3) no evidence manifest exists (`omni-evidence-v1.json`) — `qf_corpus` defines the schema, nobody has written the builder; (4) C8 residual-width not implemented; StreamedHarness for 27B not implemented (`capture_grams` exists, untested); (5) GGUF hook and frozen-review hook are dry runs; (6) C1 untested above 1.33x and without sidecar regeneration; (7) untested paths: `reader_metric` arm, `--gdn-pairing`, `--gdn-drop-arm`, per-layer termgain, tied-vocab branch.

ENTRY POINTS:
```bash
# self-test suite (CPU, ~90 s, 16 tests) — detached, then grep the log
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script 'cd /home/cereal/omni-campaign/qwen-fold && rm -rf .work && setsid nohup /home/cereal/model-surgery-lab/venv/bin/python qf_selftest.py --work /home/cereal/omni-campaign/qwen-fold/.work --json-out /home/cereal/omni-campaign/qwen-fold/selftest-results.json > run.log 2>&1 < /dev/null & sleep 3; echo started' --timeout 100
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --execute "grep -E '^(PASS|FAIL)' /home/cereal/omni-campaign/qwen-fold/run.log"
# evidence block generated FROM the artifact (only legal source of a number about this lane)
... --execute "cd /home/cereal/omni-campaign/qwen-fold && /home/cereal/model-surgery-lab/venv/bin/python qf_evidence.py"
# builds on the 4B (GPU0, bf16). One axis per child. Replace <TAG>/<flags>:
QF=/home/cereal/omni-campaign/qwen-fold; P=/home/cereal/models/Qwen3.5-4B; EV=$QF/evidence/omni-evidence-v1.json
$PY $QF/qf_build.py --parent $P --out $QF/children/QF-C2K2304 --evidence $EV --tag C2K2304 --parent-code Q35-4B --device cuda:0 --dtype bfloat16 --mlp-keep 2304     # C2 4x
$PY $QF/qf_build.py ... --tag V4X    --vocab-keep 62080                                # C1
$PY $QF/qf_build.py ... --tag X03    --excise 3                                        # C3/C4 (one layer per child)
$PY $QF/qf_build.py ... --tag G5V16K8 --gdn-value-heads 16 --gdn-key-heads 8           # C5 arm A (4B: 32/16 -> 16/8)
$PY $QF/qf_build.py ... --tag G5D32  --gdn-key-head-dim 32 --gdn-value-head-dim 32     # C5 arm B
$PY $QF/qf_build.py ... --tag N6K48  --nope-keep 48                                    # C6 (head_dim 256 -> 112)
$PY $QF/qf_build.py ... --tag K7KV2  --kv-heads 2 --kv-arm kv-only                     # C7 kv-only
$PY $QF/qf_build.py ... --tag K7G2   --kv-heads 2 --kv-arm full-group                  # C7 full-group
$PY $QF/qf_build.py ... --tag T9     --terminal-gain                                   # C9
# queue wrapper (GPU0 preflight "8.6 NVIDIA GeForce RTX 3090" + digest reproduction check)
cd $QF && ./bodyq.sh C2K2304 --mlp-keep 2304
# GGUF hook (real once a llama.cpp checkout + gguf-py exist; export lane has one at /home/cereal/omni-campaign/export/llamacpp)
$PY $QF/qf_build.py ... --gguf /home/cereal/omni-campaign/export/llamacpp --gguf-out $QF/children/QF-C2K2304.gguf
```

### 1.2 ledger — registry, clone/delete, packages, dashboards (`OC/ledger/`, `OC/CHECKPOINT_POLICY.md`)

Modules: `oc_seal.py, oc_ledger.py, oc_checkpoint.py, oc_package.py, oc_dashboard.py, test_oc_ledger.py, smoke_oc.sh, README.md`. Stdlib only. Canonical JSON byte-identical to the LFM2 ledger's: reads all 1,056 ancestor rows and reproduces `d2027831e229d763…` [verified]. Parameter counts from safetensors headers, no torch: Qwen3.5-0.8B = 873,438,784; Qwen3.5-4B = 4,659,865,088 [verified]. Directory digest 105–169 MB/s (a 27B child ≈ 6–9 min) [verified].

READY: `register` (counts ALWAYS read from headers; a disagreeing typed count is refused; `--allow-asserted-parameter-counts` labels the exception everywhere), `record-review` (status COMPUTED from counts; `--status` may only assert agreement; only `--cyril-ruling` overrides), `clone` (byte copy, digest asserted equal, sealed clone receipt, `chmod -w`, status `cloned`; refuses unreviewed source, refuses protected destinations), `delete` (fourteen R4 clauses printed PASS/REFUSE, chain lock held for the whole command, record built before unlink, nothing without `--confirm`), `oc_package build/verify` (seven-section REVIEW_<TAG>.md, READ files with the ancestor's collapsed-repetition measure, MANIFEST.json re-verified on write), `oc_dashboard` (SCORECARD.md with a `ratio from` column, REVIEW_QUEUE.md), `oc_ledger merge` (append-only, prefix-checked, fsync).

GAPS: (1) round-2 blocker: `_posix_parts` uses `.absolute()` without `resolve()`/`normpath` — `--directory $OC_ROOT/../models/Qwen3.5-4B` or a symlinked root bypasses the `/mnt`,`/home/cereal/models` guard and `_assert_directory_is_free`; fix = resolve before matching and seal the resolved path; (2) ARCH/behaviour sealers (`seal_matrix_architecture_child.py`, `run_lfm2_composition_frozen_review.py`) still hardcode `Lfm2ForCausalLM` — no omni cell rows yet, only checkpoint rows; (3) `bands.json` per parent does not exist; (4) `--parent-directory` unguarded (cheap verify can walk the share if it points there); (5) verify raises instead of listing a PROBLEM on a symlinked subdirectory; (6) media `audio_rms` rule only bites when the runner supplies an rms.

ENTRY POINTS (server; `export OC_ROOT=/home/cereal/omni-campaign; L=$OC_ROOT/ledger/oc_checkpoint.py`):
```bash
timeout 250 python3 $OC_ROOT/ledger/test_oc_ledger.py -v                       # 95 tests
timeout 270 bash $OC_ROOT/ledger/smoke_oc.sh                                    # 16-step CLI smoke
# register a fold child (dry run first: drop --confirm)
$PY $L register --checkpoint-id OC01-F-Q35-4B-C2K2304 --stage fold --directory $OC_ROOT/oc01/children/OC01-F-Q35-4B-C2K2304 --parent-directory /home/cereal/models/Qwen3.5-4B --parent-code Q35-4B --parameter-count auto --parent-parameter-count auto --recipe-command "qf_build.py ... --mlp-keep 2304" --component "mlp:<parent_params>:<child_params>:model.language_model.layers." --confirm
$PY $L record-review --checkpoint-id OC01-F-Q35-4B-C2K2304 --review-json review-C2K2304.json --confirm
$PY $OC_ROOT/ledger/oc_package.py build --checkpoint-id OC01-F-Q35-4B-C2K2304 --out $OC_ROOT/oc01/human-review/OC01-F-Q35-4B-C2K2304 --dossier <dossier.json> --per-case <per-case.json> --canonical-reread <canonical-reread-oc01.json> --receipt <cell-receipt.json> --verdict <verdict.json> --build-receipt <qf-build-receipt.json> --bench <bench.json> --media <media/>
$PY $OC_ROOT/ledger/oc_package.py verify --package $OC_ROOT/oc01/human-review/OC01-F-Q35-4B-C2K2304
$PY $L clone --source OC01-F-Q35-4B-C2K2304 --stage-about-to-run D --confirm    # BEFORE any training
$PY $L delete --checkpoint-id <id> --reason "<paragraph>" --rebuild-command "<exact>" --retained-source <id> --confirm
$PY $L verify ; $PY $L verify --deep ; $PY $OC_ROOT/ledger/oc_ledger.py verify --ledger $OC_ROOT/ledger.jsonl --registry $OC_ROOT/registry
$PY $OC_ROOT/ledger/oc_dashboard.py                                             # SCORECARD.md + REVIEW_QUEUE.md
```

### 1.3 k3 — streamed harness + C14 expert-union coreset (`OC/k3/`, `/home/cereal/omni-campaign/k3/harness/`)

Modules: `k3_common, k3_layer, k3_stream, k3_receipt, c14_union_coreset, k3_review_stream, k3_shapes, k3_synth, gram_memory_probe, test_k3_harness`, README with a receipts table (nine receipts in the lane dir, hash-identical to `/home/cereal/omni-campaign/k3/ship/`). K3 modeling source copied to `/home/cereal/omni-campaign/k3/k3_src/`.

READY [verified on synthetic K3-shaped checkpoints]: MXFP4 dequant bit-exact vs compressed_tensors; MoE bit-exact vs `KimiSparseMoeBlock` (relL2 0.0, top-k agreement 1.0); whole layer exact at fp32 (1.1e-7) and 5–7e-3 at bf16 (proposed D-a2 line 2e-2); AttnRes residual PAIR (prefix_sum + block snapshots) transcribed and checked on both branches; fla `chunk_kda` vs pure-torch recurrence 6.2e-3 on the free 3090; C14 two-stage mass-allocated pivoted selection with the λ ladder in Gram form, form E/form M, router deleted; device budget now prices resident terms (29.5x under-pricing fixed; `--experts-device` defaults to cpu); R-proxy (teacher-forced, any N) and decode (only when every MoE layer is folded) review modes, baseline provenance stamped.

GAPS: (1) round-2 blocker: `LayerWeights.device_bytes` and `CoresetFold.device_resident_bytes` compare `t.device == device` and `torch.device('cuda') != torch.device('cuda:0')` — normalise the index; (2) **no real K3 layer has run through the harness** (only headers of layers 1 and 3 read); (3) **D-a2b unresolved**: shipped `A_log` is F32[128] = head_dim while the modeling code declares [num_heads=96] and the repo module refuses its own weights; the import lane trims 128→96 ("checkpoint zero-pads to head_dim") and gets layer-0 exactness 3.47e-8 against the reference — that is evidence for the trim reading, but the harness resolves the axis by shape and the two lanes must agree on ONE reading before any calibration cache is trusted; (4) host `gram_memory_estimate` omits prior copies and is uncalibrated vs its one anchor (3.087 vs 3.814 GB measured); (5) K=32,768 and 65,536 never run; chunk_kda never at 96×128 geometry; no tokenizer (random ids only); `--resume` overwrites token provenance.

ENTRY POINTS:
```bash
K=/home/cereal/omni-campaign/k3; H=$K/harness; M=/mnt/ai_data/models/moonshotai/Kimi-K3
timeout 3000 $PY $H/test_k3_harness.py --work-dir $K/testwork5 --src-dir $K/k3_src --out $K/k3_harness_test_receipt.json   # 29 tests (detached)
timeout 180 $PY $H/k3_shapes.py --model-dir $M --layer 1 --pattern self_attn --out $K/k3_layer1_shapes.json                # header census
# K3-1 exactness on the REAL layer 1 (reads 16.99 GB from the share once; --reference-moe needs 59.2 GB RAM)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<3090 UUID> timeout 7200 $PY $H/k3_receipt.py --model-dir $M --src-dir $K/k3_src --layer 1 --tokens 256 --device cuda:0 --kda-backend fla --reference-moe --out $K/k3_layer1_exactness_receipt.json
# streamed calibration pass (per-layer cache + body staging + snapshots + resume)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<UUID> timeout 280 $PY $H/k3_stream.py --model-dir $M --layers 0-92 --random-tokens 65536 --tokens-tag 65536 --cache-dir $K/cache --body-dir $K/body --device cuda:0 --kda-backend fla --cache-set minimal --snapshot-every 12 --resume   # detached; poll
# C14 on layer 1 (experts on cpu by default; four rungs)
timeout 280 $PY $H/c14_union_coreset.py --model-dir $M --layer 1 --cache $K/cache/T65536/L01/tensors.safetensors --out-dir $K/folds --rungs 65536,32768,17408,8704 --k-pool-factor 1 --device cuda:0 --experts-device cpu
# review: parent baseline (refuses if anything folded), then child vs baseline (refuses a folded baseline)
timeout 280 $PY $H/k3_review_stream.py --model-dir $M --mode proxy --case-file cases.json --emit-baseline $K/parent_logits.safetensors --out $K/review_parent.json
timeout 280 $PY $H/k3_review_stream.py --model-dir $M --mode proxy --case-file cases.json --dense-dir $K/folds --rung 17408 --baseline $K/parent_logits.safetensors --out $K/review_child.json
timeout 1800 $PY $H/gram_memory_probe.py --out $K/k3_gram_memory_probe_receipt.json
```

### 1.4 media — C15 VAE absorption, C16 ACE-Step DiT transplant, H3-0 (`OC/media/`, `/home/cereal/omni-campaign/media/`)

Package `mediafold/{linalg, receipts, vae_adapters, c15_pairs, c15_fit, c15_video_groups, c15_merge, ace_dit, c16_capture, c16_transplant}` + scripts `c15_vae_probe, c15_vae_pairs_container, c15_video_groups_container, c15_absorb_run, c16_transplant_run, h3_0_proof_rung, run_tests`. Backup `/home/cereal/omni-campaign/media_code_backup_20260902-035606.tar.gz`. `receipts/retractions.json` records withdrawn rows.

READY / RESULTS [verified, out/ receipts]: real-VAE pair corpora — ACE 1.5 (stacked, 3840↔64, 9,600 rows/96 clips), H3 audio (800↔32 ctx0 and 4000↔32 ctx2, 7,680 rows each), H3 video (measured tubelet, 3072↔24, 9,600 rows; two chunks proven to agree on the ownership map). Ladder (fp64, held out by clip): ACE linear 23.09 dB vs parent round-trip 32.78 (gap −9.69), PCA rank-64 bound 36.42 dB → `ceiling_below_linear_bound=true` (a linear map of that width provably reaches the parent; the fit is an UNDER-FIT, not a ceiling); H3 audio 24.13 vs 36.34, bound 32.76 (3.58 dB BELOW parent → genuinely needs a companion decoder or a latent-speaking band); H3 video encoder cos 0.984 but the parent's own round-trip on the 64 px procedural corpus is 9.91 dB (verdict flagged uninformative). C16 block 12 driven from real layer-12 activations via `--prefix-shell` (delta/input 0.138), rope threaded, norm-matched control, MLP coreset exactly 4.00x (37,748,736 → 9,437,184; whole block 1.82x), honest holdout rel_l2 0.617 / cos 0.812; replace vs control 0.0628 vs 0.0488 through a chart the receipt itself rates as noise (forward r2 −0.007). H3-0 v5: raw 11/13, absorbed 12/13, control 11/13 (one gate, one seed). Measured audio VAE temporal support = 5 frames (±2) for both audio VAEs.

GAPS: (1) round-2 blocker: `coreset_block_mlp` groups by contiguous 256-row batch slices, not by clip (100 rows/clip) → ~39% of held-out rows share clips with fit; carry `clip_index` through `latent_stimulus` and group by clip, re-run; (2) ACE under-fit undiagnosed — D is fitted on the true latent, never on E(raw); one-line closed-form experiment not run; (3) H3 audio frame↔latent ownership at the head is non-identity per the lane's own probe ([0,0,2,4,4,5,…]) — pair with a measured head like the video path; (4) no ACE ctx2 row; (5) H3-0 still on synthetic VAEs; separation not seed-swept; (6) whole-block 4x needs the kv-head merge (rung 2); (7) Hunyuan3D ShapeVAE and Qwen-Image VAE never exercised; 3D needs the D10 voxel-SDF anchor pairing; (8) no decoded audio from any transplant, nothing blind-scored by Cyril.

ENTRY POINTS:
```bash
MD=/home/cereal/omni-campaign/media
timeout 400 $PY $MD/run_tests.py --ace-weights /home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors --chimera-src /home/cereal/chimera-proof-c5a5c18-v4    # 56 tests
# VAE pairs in OUR container (VAE only; no DiT, no H3 generation). CPU for audio, GPU0 for video:
timeout 5400 docker run --rm --entrypoint /opt/environments/python/comfyui/bin/python -v /home/cereal/comfyui/models:/opt/ComfyUI/models -v /home/cereal/omni-campaign:/work comfyui:codex-minimax-h3-runtime-env-r5-20260808 /work/media/c15_vae_pairs_container.py --vae /opt/ComfyUI/models/vae/ace_1.5_vae.safetensors --out /work/media/pairs/ace_s_ctx2_c0 --clips 16 --seconds 4.0 --context 2 --batch 1 --device cpu --path-mode direct --pairing stacked --seed 20260901
cd $MD && ./run_ace_stacked.sh                              # six ACE chunks + strict merge
$PY -m mediafold.c15_merge --out pairs/<merged> --pairs pairs/<c0> --pairs pairs/<c1>     # refuses on a map mismatch
$PY $MD/c15_absorb_run.py --out out/c15_real_v4 --hidden 5120 --rungs linear,nonlinear --pairs ace_1.5_vae_stacked=pairs/ace_s_ctx0_merged --pairs h3_audio_ctx0=pairs/h3a_d_ctx0_merged --pairs h3_audio_ctx2=pairs/h3a_d_ctx2_merged
$PY $MD/c16_transplant_run.py --out out/c16_L12_v3 --block 12 --rung latents --pairs pairs/ace_s_ctx0_merged --batch 36 --length 256 --host chimera --chimera-src /home/cereal/chimera-proof-c5a5c18-v4 --host-layer 3 --host-batch 128 --keep 1536 --chart-methods ridge,svd,random --device cpu --prefix-shell
CUDA_VISIBLE_DEVICES=0 $PY $MD/h3_0_proof_rung.py --chimera-src /home/cereal/chimera-proof-c5a5c18-v4 --out out/h3_0_v6 --arms raw,absorbed,control --steps 600 --device cuda
timeout 1800 docker run --rm --entrypoint /opt/environments/python/comfyui/bin/python -v /home/cereal/comfyui/models:/opt/ComfyUI/models -v /home/cereal/omni-campaign:/work comfyui:codex-minimax-h3-runtime-env-r5-20260808 /work/media/c15_vae_probe.py --vae /opt/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors --out /work/media/out/vae_probe_support --device cpu --path-mode direct --clips 2 --seconds 1.0 --audio-support-frames 12
```

### 1.5 export — GGUF, parity, 256k envelope (`OC/export/`, `/home/cereal/omni-campaign/export/`)

Files: `llama_probe.py, envelope_bench.py, kv_fit_search.py, parity_check.py, export_hf_to_gguf.py, fa_negative_control.py, p40probe_check.py, wait_free_vram.sh, rerun_parity_iq4xs.sh, compare_parity_passes.py, build_llamacpp_cuda.sh, run_e2e_0p8b.sh, test_export_lane.py, EXPORT_LANE_SPEC.md`, receipts under `receipts/{envelope,parity,probe,fa-control}`. Own llama.cpp build at `/home/cereal/omni-campaign/export/llamacpp/bin` (commit `b96806d9`, sm61+75+86, llama-server sha `233c52ce…`), no `llama-imatrix`.

RESULTS [verified]: full 27B×256k envelope on the 3090 (receipt `6626c991…`, complete): 4,096 / 32,768 / 65,536 / 131,072 / 260,712 tokens → prefill 1,279 / 1,161 / 949 / 788 / 560 tok/s, decode 38.9 / 32.8 / 22.3 / 20.6 / 12.4 tok/s, 12/12 needles hit, own peak 21,522 MiB. q8_0/q8_0 KV at 256k does NOT fit 24 GB (measured OOM); it loads at 233,472 and fails at 237,568 on a P40 (`71c59eca…`). IQ4_XS parity-checked (mean char prefix 27.4 vs Q4_K_M 43.9) with provenance; KV cache TYPE changes 10/15 greedy answers (f16 vs q4_0) — same-config passes are 0/15 different. Container `p40probe`/`p40_256k`/`gpu0_256k` no longer exist (receipt `df3a1597…`). Phase-0 P40 envelope (server-cuda image): 250,940 tokens, needles 16/16 across both cards, P40 ~8x slower at depth.

GAPS: (1) imatrix not plumbed — build `llama-imatrix` into a SEPARATE root `llamacpp-imatrix` (never overwrite `llamacpp/bin` while a sweep runs); (2) the P40 27B sweep `oc-env-27b-gpu1-q4kv` is still in flight on GPU1 running the PRE-FIX module — read its `points` array, not its `partial` flag; (3) `kvfit-…-q8kv-p40gpu3.json` predates the diagnostic fields; (4) `--runner docker`, parity HF side at 27B, media-output export (D5b), contamination screening of needle prompts all unexercised; (5) spec/receipts still name GPU4 for some earlier runs.

ENTRY POINTS:
```bash
E=/home/cereal/omni-campaign/export
timeout 240 $PY -m pytest $E/test_export_lane.py -q                                                        # 145
$PY $E/export_hf_to_gguf.py --hf-dir /home/cereal/models/Qwen3.5-4B --out-dir $E/gguf --tag Q35-4B-stock --outtype bf16 --quants Q4_K_M,IQ4_XS
$PY $E/parity_check.py --tag <TAG> --server-url http://127.0.0.1:18097 --gguf-path <gguf> --hf-dir <hfdir> --out-dir $E/parity      # measured HF column
$PY $E/envelope_bench.py --tag OC-<child>-gpu0-q4kv --model <gguf> --runner host --gpu-uuid GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561 --gpu-index 0 --kv q4_0 --ctx 262144 --port 18090 --prefill-points 4096,32768,131072,262144 --needle-points 32768,65536,131072,262144 --needle-depths 0.1,0.5,0.9 --max-wall-s 10800 --r3-control $E/fa-control/fa-negative-control.json
$PY $E/kv_fit_search.py --tag <TAG> --model <gguf> --kv q8_0 --gpu-uuid <UUID> --gpu-index <i> --lo 8192 --hi 262144 --port 18096 --load-timeout 420
$E/wait_free_vram.sh --gpu-index 0 --min-free-mib 21000 --deadline-s 10800 --poll-s 30 -- <command>
cd $E && setsid nohup ./build_llamacpp_cuda.sh --arch "61;75;86" --jobs 44 --root $E/llamacpp-imatrix > llamacpp/logs/build-imatrix.log 2>&1 < /dev/null &   # [assumed --root flag; check the script before use]
# watch the in-flight P40 sweep
python3 -c "import json;d=json.load(open('/home/cereal/omni-campaign/export/envelope/envelope-OC-Q36-27B-Q4KM-gpu1-q4kv.partial.json'));print([p['target_tokens'] for p in d['points']])"
```

### 1.6 calibration — parent bank runs, bank v2 draft, assessments (`OC/calibration/`, `/home/cereal/omni-campaign/calibration/`)

Files: `oc_calib.py, oc_run_frozen_bank.py, oc_assess.py, oc_parent_calibration.py, oc_bank_v2_draft.py, oc_testcards.py, oc_kernel_probe.py, test_oc_calibration.py, run_r6.sh, run_r7.sh, README.md`, `banks/` (v1 pack `0ee6bf14…`, probe/smoke packs, bank v2 DRAFT `f26f36bf…` byte-reproducible with 5 PNG cards), `out/` dossiers + assessments + scoring sheets, `out/KERNEL-FIELD-CORRECTION-2026-09-02.md`.

RESULTS [verified]: Qwen3.5-4B loaded as `Qwen3_5ForConditionalGeneration` (4,539,265,536 live params, dir digest `94cb094a…`). Template-default bank v1 lane digest `3a13002f…` reproduced byte-identically across r5 and r7 (16/16 lanes; 13 cap-rides at 1,024 tokens because the template opens a think block) — r7 correctly labels it **NEW ANCHOR** because r5's validity tuple has four unmeasured elements; `out/oc-dossier-Q35-4B-REPEAT.json` is the anchor with the full v2 tuple. `--thinking off` surface: 16/16 complete, 643 s, column-B upper bound 13/15 (dossier `aa75b45b…`, still v1 kernel semantics). Bank v2 text lanes decoded; long-context: 8k (~10.2k real tokens) 3/3 complete with the pinned needle exactly; 32k (41,563) answered; 64k (~84,100) 3/3 complete after the KV-cache fix; 200k (~258,800) genuine OOM in bf16 HF on 24 GB, boxed as artifact. Kernel path measured off the loaded model: `chunk=fla-0.5.2; recurrent=fla-0.5.2; conv-prefill=torch; conv-decode=torch`.

GAPS: (1) NOTHING GRADED — every column is `is_upper_bound: true`; Cyril's per-case files are the missing input; (2) `--thinking off` dossier needs one ~11-minute re-run to get a v2 tuple; (3) vision (5 cards) and AV (5 spec) lanes never decoded — no processor/asset loader; (4) long-context lane ids under-count real tokens by ~25–30% (draft bank digest would move if re-cut — Cyril's call); (5) copy measure (0.60 union coverage) reads think blocks as ECHO — retune is an open decision; (6) `load_bank`/`compute_architecture`/`generate_greedy` pinned to sealed originals only by hand-diff; `logits_to_keep` A/B never run; (7) `bands.json` (PARENT_REVIEW_BAND for Q35-4B) cannot exist until Cyril scores the parent.

ENTRY POINTS:
```bash
C=/home/cereal/omni-campaign/calibration
timeout 240 $PY $C/test_oc_calibration.py                                                   # 77
timeout 250 $PY $C/oc_kernel_probe.py --model-dir /home/cereal/models/Qwen3.5-4B --device cpu --dtype bfloat16 --out $C/out/kernel-probe-v2-Q35-4B.json
# parent calibration, one arm (GPU0, bf16, sm86 asserted); --thinking off is the readable surface
$PY $C/oc_run_frozen_bank.py --model-dir /home/cereal/models/Qwen3.5-4B --label Q35-4B-NOTHINK --parent-code Q35-4B --stage parent --device cuda:0 --require-compute-architecture sm86 --dtype bfloat16 --maximum-new-tokens 1024 --thinking off --bank banks/lfm2_semantic_whole_response_prompts_v1.json --probe-prompts banks/matrix_campaign_probe_prompts_v1.json --smoke-prompts banks/matrix_campaign_smoke_prompts_v1.json --out out/oc-dossier-Q35-4B-NOTHINK.json
# a CHILD on the same surface (stage fold): same command with --model-dir <child dir> --label <TAG> --stage fold
# bank v2 text lanes / long-context rungs on a model
$PY $C/oc_run_frozen_bank.py ... --thinking off --bank banks/omni_review_bank_v2_draft.json --lanes <ids> --text-only --out out/oc-dossier-b2-<label>.json
cd $C && nohup setsid ./run_r7.sh /home/cereal/models/Qwen3.5-4B Q35-4B cuda:0 > out/run-r7-Q35-4B.log 2>&1 < /dev/null &   # 64k, 200k, REPEAT; refuses a busy card
$PY $C/oc_assess.py --dossier out/oc-dossier-<label>.json --bank banks/<bank>.json --out-json out/assessment-<label>.json --out-markdown out/SCORING-SHEET-<label>.md [--per-case out/per-case-<label>.json]
& "C:/Python314/python.exe" "C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/calibration/oc_bank_v2_draft.py" --out banks/omni_review_bank_v2_draft.json --assets-dir banks/assets --v1-pack "C:/Users/cereal/Documents/ai training/neural-forge/research/lfm2_semantic_whole_response_prompts_v1.json" --self-test
```

### 1.7 chimera-import — S1–S5 on branch `claude/omni-import` (reports in `OC/chimera-import/`)

READY [verified, each stage reviewed to pass]: config switches (KDA grouped keys, GDN decay/gate/qk-norm modes, SiTU inf, AttnRes identity bias, tied embeddings) with defaults bit-identical to `c1ca0b2`; `gated_gqa` block (vs real `Qwen3_5Attention` 1.19e-6 at 0.8B geometry); per-layer `intermediate_sizes`; fixed-band ledger (248,320→252,416; 163,840→167,936) with band logit mask; Qwen JSON-BPE + K3 tiktoken readers, token alignment + embedding bridge (control ids unmapped); native `volume` type (id 7, 4096); import profiles reading geometry from the checkpoints; safetensors sharded IO; `import_qwen35_state` (R-Q Qwen3.5-0.8B cpu/fp32 max abs logit diff 3.19e-5, argmax 210/210; bf16 GPU6 column 209/210); `import_k3_layer_state` (R-K3 layer 0 cpu/fp32 3.47e-8, rel-L2 9.7e-7, 0 missing/unexpected keys, A_log trimmed 128→96); qwen35 GGUF writer (320 tensors, sha `fcb61be2…`) served by llama.cpp b10068 with 3/3 continuations byte-identical to HF bf16.

GAPS: MLA path (layer 3) never measured; `load_k3_moe_tensors` never executed; 4B and 27B receipts not run; spec Q1 (`output_gate_type='swish'` on the 27B; transformers 5.12.1 applies sigmoid unconditionally) unresolved; `situ(inf,inf)` not bit-equal to silu (4.77e-7); `attnres_init='identity'` is identity per AttnRes but not a plain residual (only `residual_mode='none'` is); RoPE table fp32 (~2e-4 at 262k); kimi-k3 GGUF export of a real dense-K3 child untested; recurrent-state layout `[B,H,K,V]` vs fla `[B,H,V,K]` documented only.

ENTRY POINTS: see `OC/chimera-import/IMPORT_LANE_STATUS.md` section 7 (receipt commands for S1–S5; the R-Q/R-K3 commands use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src` from the branch checkout synced at `/home/cereal/omni-campaign/chimera-omni-import/src`). Test suite: `PYTHONPATH=src $PY -m pytest -q` on the branch (376).

### 1.8 Datasets (R1) and training constraints (R5) — reference state, no code

- The curated high-value set = the master registry above. `rows_by_state`: ready_untrained_candidate 52,416; trained_base 28,025; trained_addenda_complete 4,964; trained_v37_complete 8,913 (the `trained_*` labels refer to the dead omnicoder — for Chimera every row is untrained) [verified]. Loader: `chimera_omni.data.curated_shards` (normalize_row drops `training_allowed False` / `do_not_train`). Sidecars resolve 63/64 at last check. NO ≥32k-token rows exist anywhere (longest ~12.6k chars); the long-context evidence corpus must be BUILT. The bounded checksum walk (pointer → 631 entries → shard sha256) and the per-route token census are still UNDONE.
- Training modes that fit this hardware [derived, unmeasured]: full-param AdamW at ≤2B on one 24 GB card; 4B full-param only with CPU-offloaded Adam (50 GB fp32 master/m/v in RAM); 9B–27B LoRA only (27B base 54 GB split GPU6+GPU4); QLoRA violates quantize-last (default NO, D-R5-8). 27B teacher forward ≈460 tok/s on a 3090 → precompute top-k logits, never on-the-fly. K3 as a logit teacher ≈1M tokens per streamed pass. Every tok/s number is an assumption until `tps_probe.py` (not yet written) runs.
- Reusable verbatim from neural-forge: `stage2_kd_objective.py` (forward-KL, GJSD, `Stage2DonorVocabulary` for the 276 dead Qwen rows), `stage2_training_receipt.py` (needs its own pre-registration sha — copy into `/home/cereal/omni-campaign/distill/`). Not reusable: the LFM2 carrier ABI, checkpoint layout, freerun probe, and the LFM2 self-gen corpus (wrong teacher/tokenizer).

---

### 1.z State delta — 2026-09-02 14:20Z

- **R1 delivered.** `SCORING-SHEET-Q35-4B-NOTHINK.md` (16 lanes, 15 scored, all complete; column A 15/15,
  column B 13/15 as upper bounds; ECHO flags on `logic-ordering` 0.64 and `debugging-off-by-one` 0.93,
  threshold untouched) sent to Cyril. His grades become `PARENT_REVIEW_BAND[Q35-4B]` / `bands.json`.
  Until then every Wave A child is compared on the machine columns only.
- **chimera-omni `main` = 5a87018** (fast-forward from c1ca0b2, local only, not pushed): import lane S1-S5,
  K3 MLA layer-3 mixer receipt 8.94e-08, GGUF > 2e9-element fence, KIMI_K3_EXPORT_STATUS label, Q1 cited.
  392 server tests. Codex has the evidence (relay 17e01d58, 7bb65181).
- **K3 shard 2 (layer 1) on local disk**: `/home/cereal/omni-campaign/k3/staged/model-00002-of-000096.safetensors`,
  16,990,911,504 B (size-exact). Shard 4 also staged (16,567,501,776 B). K3-1 waits only on the closeout
  k3 device-index fix.
- **llama-imatrix / llama-quantize built** at `export/llamacpp-imatrix/build/bin` (receipt written) — the
  Q4 export lane (stage E) has its importance-matrix quantizer.
- **Server python for every campaign script**: `/home/cereal/model-surgery-lab/venv/bin/python`
  (there is no `omni-campaign/venv`). `oc_assess.py --per-case` is an INPUT (a sealed reading), never an output.
- **Closeout workflow DONE (20 agents, 0 errors, 108 min).** Verdicts: qwen-fold fix (17/17, Gram hoist bit-identical,
  saving is n_v = 48x not 96x), ledger pass (101/101; paths resolved at the root, `..`/symlink aliases refused),
  k3 pass (30/30 on GPU6; `cuda` vs `cuda:0` alias closed via canonical_device; receipts now carry the resolved
  device + the literal request), media pass (C16 holdout re-measured on a clip-ALIGNED window: block-12
  rel_l2 0.5442 / cos 0.8415 on 950 rows, activation_independence true; v2/v3/v4 are NOT a trend),
  evidence-builder fix (`qwen-fold/evidence/omni-evidence-v1.json` sealed + verified: 24 batches x 4,096, 196 rows
  placed whole, 0 rows spanning batches, 1.864% eos padding, tokenizer gate enforced at consumption — the 27B does
  NOT share the 0.8B/4B tokenizer and needs its own manifest), registry-walk pass (**631/631 entries, 640/640
  shards reproduce their sha256, 94,318 rows, 0 problems** — the earlier "unreachable rows" finding was a walker
  bug; no stage-T repair decision exists). Section 1.0/1.2/1.3/1.4 rows and line ~201 are superseded by this block.
- Vocabulary hazard (evidence-builder): 248,070 defined ids in 248,320 physical = 250 dead, not 276; the PORT_SPEC
  derivation would delete the 26 added control tokens (eos/bos/vision) in C1. C1's mandatory-token logic must be
  re-derived before any C1 child is built.
- Still running: `tps_probe.py` cells. P40 q8-KV fit search on GPU3 has released the card (receipt pending check).
- **P-1 DONE** — q8_0 KV on the P40 (GPU3), stock 27B Q4_K_M: loads at `-c 233472`, OOM at `237568`
  (receipt `export/envelope/kvfit-OC-Q36-27B-Q4KM-q8kv-p40gpu3-v2.json`, sha 7a961d5c…). 8-bit KV at 256k
  needs ~10% KV headroom from the fold (C7 / C3) or stays at q4_0.
- **tps_probe (G0-12) in progress on GPU4+GPU6**: 11 receipts under `distill/tps/` so far (0.8B full L1024/4096,
  27B teacher-stream L4096 mb1/mb8, 4B cpu-AdamW L1024, 4B LoRA L4096). GPU4 is not free for G0-2/G0-4 until it
  finishes; the v2 thinking-off calibration is queued behind it.
- **0.4 DONE — parents registered** (`--root /home/cereal/omni-campaign`; registry/ + ledger.jsonl created):
  `PARENT-Q35-4B` directory_sha256 1473b188…, 4,659,865,088 header parameters; `PARENT-Q35-0.8B` 94f2c2b5…,
  873,438,784. Provenance receipts `ledger/provenance/provenance-Qwen3.5-{4B,0.8B}.json` (every HF-listed file
  reproduces its LFS sha256 or git blob id at the HF commit; `.cache` recorded, not verified). Parent registration
  form: `--parent-directory <itself> --parent-provenance-sha256 <receipt sha>`.
- **POWER CUT ~17:32Z — server rebooted.** Every detached job and both builder agents died. Survived: child
  `QF-C2K2304` + receipt (2,840,572,416 params, net 0.6258 = 1.60x; GGUF step did not run because the receipt
  looked for `export/llamacpp/convert_hf_to_gguf.py` — the checkout is `export/llamacpp/llama.cpp/`), the
  queue/register/review scripts (untested), 12 tps receipts, K3 shards 2+4. Lost: the v2 thinking-off dossier.
  Relaunch workflow: adversarial review + fix of the four scripts, then register child 1, queue on GPU0,
  v2 calibration + child-1 review chained on GPU4, R4 throughput table compiled from the receipts.
- **RELAUNCHED 20:24Z (after review+fix of the four scripts: 14+16+21+11 confirmed findings).**
  `OC01-F-Q35-4B-C2K2304` REGISTERED (dir sha 51c083a7…, 2,840,572,416 params, net_ratio 0.6096 = 1.64x vs the
  header parent, record 1e46991a…, ledger sha 785ea9c8…). Sealed caveat: `fold_accounting.unexplained
  120,599,552` = the `mtp.*` head dropped without `--drop-mtp` declared — R2 must read it as "MTP dropped";
  Wave A round 2 passes `--drop-mtp` explicitly. Queue pid 611499 on GPU0 (C2K1152 first; ~1.8 h per C2 child).
  GPU4 chain: v2 thinking-off dossier → `oc_assess` → `wavea_review.sh OC01-F-Q35-4B-C2K2304 --gpu 4`.
  `oc_run_frozen_bank.py` needs `--probe-prompts banks/matrix_campaign_probe_prompts_v1.json --smoke-prompts
  banks/matrix_campaign_smoke_prompts_v1.json` (its default path does not exist on the box).
- **2026-09-03 00:30Z ledger**: `PARENT-Q38-27BU` registered (provenance-verified against
  JonathanColetti/Qwen3.8-27B-Uncensored @ 5bb7aa90; dir sha 8f6735a4…; 27,781,427,952 header params; ledger sha
  2926d52c…). Nine 4B children registered; `C2K2304` reviewed **partially_capable** (sealed 5 / canon 5 / A 5 / B 2 of
  15 against the parent's 15/15 — the 4x MLP fold loses most lanes on the 4B); `G5V16K8` (2x) and `N48` REFUSED by the
  per-component ≥4x rule and kept unregistered. Parent v2 thinking-off dossier + sheet exist.
- **2026-09-03 01:30–02:00Z**: NAS back and share remounted (stale autofs cleared). 27B evidence manifest sealed
  (`qwen-fold/evidence/omni-evidence-27bu-v1.json`, sha 8a3422d8…, `--allow-unpinned-tokenizer` recorded). K3 shard
  map: shard i = layer i-1 (i = 1..93), shards 94-96 = non-layer tensors; `k3/k3_stage_layers.sh` staging layers
  0,2,4,5,6,7 then the non-layer shards (120 GB free-disk guard). K2KV build FAILED at layer 27 (Cholesky on a
  non-PD Gram in `prior_anchored_gram`; queue stopped by design; debugger fixing the solver and resuming
  K2KV/K2FG/T9/V62080). `wavea_review_all.sh --gpu 4` reviewing the other registered children (side measurement;
  yields GPU4 to the composite between children).
- **2026-09-03 02:05Z — Lexar SL600 3.7 TB SSD on the AI server at `/mnt/lexar` (exFAT, persistent fstab entry).**
  The whole Kimi K3 checkpoint is being copied to `/mnt/lexar/models/moonshotai/Kimi-K3` (`k3/k3_copy_to_lexar.sh`,
  layer order, seeded from local shards; ~20 h share-bound). Nothing deleted (2.2 TB free sufficed). Once complete,
  the K3 streamed pass reads from the SSD, not the share — the 16-h-per-pass share bottleneck disappears.
- **COMPOSITE ARTIFACT 1 — the 27B-Uncensored loads into the Chimera architecture exactly (2026-09-03 01:51Z):**
  `receipts/qwen38_27bu_import_receipt.json` (chimera-omni `claude/omni-import` 5b35906): cpu/float32, 210 argmax
  positions 210/210, max_abs_logit_diff 3.03e-04 against reference absmax 23.47, every one of 64 layers inside the
  1e-5 relative band; 26.90B parameters imported, vision + MTP carried as side records; the swish/sigmoid waiver and
  the Q1 observation sealed in the receipt. No Chimera-format checkpoint materialised yet (that is a run-book stage).
- **K2KV root cause (2026-09-03 05:33Z)**: layer 27 `q_norm` coordinate 1 has bf16 weight exactly −1.0, so the
  `1+w` gain is exactly 0 and C7's divide-by-gain produced Inf → NaN stream → NaN Gram → the Cholesky error. Fixed
  in `qf_attn.unfold_gain` (parent's raw row where gain is 0), `qf_solve` (named Gram defects, recorded eigen
  fallback, refuses non-finite), `_resolve_writer` (non-finite rows dropped and counted, 2% ceiling). 23/23
  self-tests; unaffected layers bit-identical. Rule for every fold class incl. the 27B: guard exact-zero norm gains.
  Queue resumed for K2KV → K2FG → T9 → V62080 (`QUEUE_TAGS`).
- **R4 table (G0-12) WRITTEN** from the 12 tps receipts: `distill/TPS_TABLE_2026-09-02.md`.
- **G0-1/G0-3 STARTED** — `qf_build.py` refuses a no-axis build, so rung 0 is carried by the first child's own
  parity + causal receipts. `C2K2304` building on GPU0: parity deterministic, logits bit-identical, hook placement
  worst relL2 3.37e-03; C2 fit relL2 per layer 0.33–0.48 at 4x. `LLAMA_CPP` must be
  `/home/cereal/omni-campaign/export/llamacpp` (`/home/cereal/llama.cpp` does not exist). Queue + register bridge
  (`wavea_queue.sh`, `qf_register.py`) being built for the remaining 14 children.

### 2.pre RESEQUENCED 2026-09-02 ~21:00Z — Cyril's ruling: STEP 1 IS THE COMPOSITE FOLD

Cyril: "[Cyril quote withheld]" Answer: no — the ledger held two parents and one single-class 4B child. From now on:

- **Step 1 = ONE registered dense Chimera Omni child containing ALL seven sources** (Qwen3.8-27B-Uncensored trunk;
  K3 folded dense and bridged; ACE-Step, H3, Qwen-Image, Hunyuan3D 2.1, Qwen3-VL as native media bands), exported to
  one checkpoint and reviewed per modality. Its run book: `COMPOSITE_STEP1_RUNBOOK.md` (+ `COMPOSITE_STEP1_STAGES.json`).
- The Wave A single-class ladder on the 4B (section 2.1 G0-3/G0-4) is **side measurement only**; it keeps GPU0 but
  yields to the composite on every other resource (GPU4, GPU6, RAM, share, disk).
- Every status report leads with: does the composite child exist, and which sources it contains.
- **Run book WRITTEN 2026-09-03 02:05Z**: `COMPOSITE_STEP1_RUNBOOK.md` / `COMPOSITE_STEP1_STAGES.json` — child
  `OC02-F-Q38-27BU-OMNI1` (~27.29B), 13 stages, nine new tools under `composite/tools/`. S0 started (parent show +
  deep verify). The 27B exact-import receipt (S0's exactness gate) already holds (5b35906). S2's share pricing is
  superseded by the Lexar SSD copy.
- **K3 ruling (Cyril, same day): "[Cyril quote withheld]"** The K3 stage is therefore: expert-union
  coreset over ALL 896 experts for ALL 93 layers, streamed in order, each layer fitted on the FOLDED-PREFIX
  activations (teacher + folded hidden-state streams) against the teacher's routed output; then the whole dense K3
  chart-merged across the Chimera trunk layers; judged only on the whole model. G0-6 "K3-1 exactness on layer 1"
  and G0-7 "three study layers" are DELETED as gates (they remain engineering diagnostics at most).

## 2. Execution order for the next 72 hours, serialized by GPU

Clock zero = 2026-09-02 ~08:00 local. Hardware assumptions at T0 [verified at the lanes' last look, re-check with `nvidia-smi` before every launch]: GPU0 3090 free (24,101 MiB at the r7 launch); GPU1 P40 held by the in-flight `oc-env-27b-gpu1-q4kv` sweep (leave alone until it writes its receipt); GPU2/3/5 P40s free for short tests; GPU4 3090 physically idle but Codex-reserved; GPU6 RTX 8000 idle, Codex-owned. RAM 251 GiB (K3 `--reference-moe` needs 59 GB; leave 100 GB for the C14 Grams). Disk 537 GB local.

### 2.0 Before anything else (T0 → T0+2h, CPU + SSH only, no GPU)

| # | Action | Owner | Output |
|---|---|---|---|
| 0.1 | Land the four round-2 blocker fixes: qwen-fold Gram hoist (qf_gdn.py:385/458), ledger path normalisation (`_posix_parts` → `resolve()`, `_assert_directory_is_free`), k3 device-index normalisation (`device_bytes`, `device_resident_bytes`, `default_device_budget_gb`), media clip-grouped holdout (`latent_stimulus` carries `clip_index`; `c16_transplant_run.py:345`). Re-run each suite (16 / 95 / 29 / 56) and re-sync hashes. | each lane | new selftest/receipt JSONs, hash parity |
| 0.2 | Confirm the GPU4/GPU6 claims through the relay (`agent-comms` skill): the plan's D14 says both are to be claimed; memory records `gpu-slot-6` GRANTED and the import lane already ran on GPU6; Phase-0 says "all seven GPUs allocated". Record the relay reply in `OC/OPEN_ITEMS.md`. Until recorded: GPU0 + P40s only. | orchestrator | relay thread id + decision |
| 0.3 | Write the evidence-manifest builder `qf_evidence_build.py` (schema `neural-forge-omni-evidence-v1`, `qf_corpus.py`): 24 sealed batches × 4,096 Qwen tokens from registry text/code/math/tool/reasoning rows (`curated_shards.iter_master_entries`, `normalize_row`), fit/hld by batch index with the every-3rd-from-the-3rd internal check, contamination screen against bank v1 `0ee6bf14…`, bank v2 draft `f26f36bf…`, extension v2, paraphrase triplets, probe/smoke packs (≥48-char containment + digest). Tokenizer sha recorded. Write to `/home/cereal/omni-campaign/qwen-fold/evidence/omni-evidence-v1.json`. **This is the single missing tool that blocks Wave A.** | qwen-fold | manifest + `manifest_sha256` |
| 0.4 | Register the parents in the ledger: `PARENT-Q35-08B`, `PARENT-Q35-4B` (`--stage parent`, headers-derived counts, directory digests; ~1–2 min each). Do NOT register the 27B yet (6–9 min digest; not needed this window). | ledger | registry records + first ledger rows |
| 0.5 | Start the bounded registry checksum walk + per-route token census as a detached CPU job (`timeout`-wrapped per-entry reads, never `find`/`du`): pointer → 631 `SHARD_INDEX_POINTER.json` → `shards.jsonl` → sha256 per shard file vs `shard_index_sha256`; Qwen-tokenize each row. Expect hours at share speed; it feeds stage T arithmetic, not this window. | datasets | `OC/datasets/registry-walk-2026-09-02.json` |
| 0.6 | Stage K3 shard 2 (layer 1, 16.99 GB) to `/home/cereal/omni-campaign/k3/staged/` with the offset-sorted reader (11 min at 25 MB/s clean, 53 min contended). Start detached now; it gates K3-1. Do not stage more than the two shards this window needs (disk). | k3 | staged shard + MB/s receipt |

### 2.1 GPU0 (RTX 3090) queue — strictly serial, one job at a time

| Slot | Window | Job | Command (abbrev.; full forms in section 1) | Done when |
|---|---|---|---|---|
| G0-1 | T0+2h → +3h | **qwen-fold rung 0 on the real 4B**: architecture map + closed-form param count vs loader count; `harness_parity` (determinism, hook placement, logits exact); `assert_causal_masks` on the unmodified parent, on a block-0-excised live model (0), and the unrepaired negative control (nonzero); identity-gate export `--hidden-size 2560 --basis identity --ff 9216 --no-fold-solve --identity-gate strict` → byte-identical; `--kv-identity-gate --kv-heads 4` and `--gdn-value-heads 32 --gdn-key-heads 16` byte-identical. | `qf_build.py --parent /home/cereal/models/Qwen3.5-4B --evidence $EV --device cuda:0 --dtype bfloat16 ...` per rung; receipts under `qwen-fold/rung0/` | six receipts, all digests reproduced; the two bugs the fixture could not see (value-side repair on a structured parent) get their first before/after numbers |
| G0-2 | +3h → +3.3h | **calibration `--thinking off` re-run on the 4B** so the readable surface has a v2 validity tuple (11 min). | `oc_run_frozen_bank.py ... --thinking off --label Q35-4B-NOTHINK-v2` then `oc_assess.py` | `oc-dossier-Q35-4B-NOTHINK-v2.json`, lane digest compared to `bc1080a6…` with both tuples printed |
| G0-3 | +3.3h → +9h | **Wave A single-class children on Qwen3.5-4B** (build ≈ minutes each on bf16; export + stock re-load + `oc_checkpoint register` each). Order: C2 `--mlp-keep 2304` (4x) → C2 `--mlp-keep 1152` (8x) → C3/C4 singles: full layers 3, 15, 31 and linear layers 0, 8, 16 (six children; not all 32 this window) → C5 A `--gdn-value-heads 16 --gdn-key-heads 8` → C5 B `--gdn-key-head-dim 32 --gdn-value-head-dim 32` → C6 `--nope-keep 48` → C7 `--kv-heads 2 --kv-arm kv-only` → C7 `--kv-heads 2 --kv-arm full-group` → C9 `--terminal-gain` → C1 `--vocab-keep 62080` (last; 4x on a tied 248,320 table, sidecar regeneration exercised for the first time). ≈15 children. Each build's receipt names `parameters_removed`, per-component ratio (≤0.25 asserted at register), every `*_frame_mismatch`. | `bodyq.sh <TAG> <flags>` then `$L register --checkpoint-id OC01-F-Q35-4B-<TAG> --stage fold ... --confirm` | 15 registered checkpoints under `$OC_ROOT/oc01/children/`, SCORECARD rows with `ratio from: headers` |
| G0-4 | +9h → +15h | **Review each Wave A child on the readable surface**: bank v1 `--thinking off` (≈11 min/child) + bank v2 text lanes `--lanes code-rle-encode-A,reason-arith-two-step-A,lc-8k-notice-select-A,lc-8k-notice-select-B,mt-dietary-correction-B` (≈3 min); `oc_assess` → upper-bound columns; `record-review` with `is_upper_bound` counts; `oc_package build`; `oc_dashboard`. Parent dossier `Q35-4B-NOTHINK-v2` is the side-by-side. Bench row (`lm_eval` arc_easy/hellaswag/piqa/winogrande/lambada/wikitext, `--limit 200 --seed 1234`) parent FIRST, then children, if the envs at `/home/cereal/ai-benchmark-lab/envs/lm-eval` load a qwen3_5 checkpoint [assumed]. | `oc_run_frozen_bank.py --model-dir $OC_ROOT/oc01/children/OC01-F-Q35-4B-<TAG> --stage fold ...`; `oc_package.py build ...`; `oc_dashboard.py` | `REVIEW_QUEUE.md` populated; **Cyril review R2** package ready at ≈T0+16h |
| G0-5 | +15h → +17h | **GGUF export of the top-3 Wave A children by upper-bound column B** through the export lane (`export_hf_to_gguf.py` → Q4_K_M; `parity_check.py` with a MEASURED HF column on the child; envelope at `--prefill-points 4096,32768 --needle-points 32768`). First real test of `convert_hf_to_gguf` on non-stock `intermediate_size` / `layer_types` / GDN head counts (T6-B). | see 1.5 | export/parity/envelope receipts per child, or the exact refusal text from the converter |
| G0-6 | +17h → +19h | **K3-1 exactness on the real layer 1** (needs the staged shard from 0.6, 59 GB RAM for `--reference-moe`, 3090 for fla `chunk_kda`): `k3_receipt.py --layer 1 --tokens 256 --reference-moe`; then the A_log disambiguation: `load_state_dict(strict=True)` into `KimiDeltaAttention(KimiLinearConfig(**text_config), 1)` AND the fla-vs-Chimera-recurrence comparison under BOTH broadcasts, alongside the import lane's trim-to-96 reading (its layer-0 receipt is 3.47e-8 under the trim). | see 1.3 | `k3_layer1_exactness_receipt.json`: prefix_out relL2, top-16 agreement, timings (MB/s, dequant s, attention s, MoE s), and a written A_log verdict for **Cyril R3** (D-a2b) |
| G0-7 | +19h → +27h | **K3 streamed calibration on the three study layers** (early 1, mid 45, late 89 — [proposed]) at T=65,536 random ids (no tokenizer yet), minimal cache set, experts packed on the card, body staged. Per layer: one 16–17 GB shard read (11–53 min) + minutes of compute. Then **C14 on layer 1** at rungs 17,408 and 8,704 only (stage 2 to K_pool=2K; host Gram 11.7 GB / 3.4 GB — the 65,536 rung stays unrun), form E + form M, receipts. If GPU6 is claimed by then, move this slot there (2.3). | `k3_stream.py --layers 1,45,89 ...`; `c14_union_coreset.py --layer 1 --rungs 17408,8704 --experts-device cpu` | `cache/T65536/L{01,45,89}/`, `folds/L01/K{17408,8704}/`, receipts with `veto_binding`, heldout y_lat relL2 — the first C14 numbers on real experts |
| G0-8 | +27h → +31h | **K3 R-proxy** on layer 1 folded (baseline emitted with nothing folded, then child vs baseline) over bank v1 + bank v2 text prompts with reference continuations — teacher-forced across all 93 layers streamed ONCE (share-read bound: ~16 h clean, ~3 days contended [verified Phase-0]). **Start detached here; it will not finish inside 72 h at contended share speed.** Diagnostic only. | `k3_review_stream.py --mode proxy ... --emit-baseline`, then `--dense-dir folds --rung 17408 --baseline` | partial receipts polled; complete receipt lands after the window |
| G0-9 | +31h → +36h | **Media on GPU0** (short, containerised, VAE only): ACE `--context 2` stacked pairs (6 chunks, CPU in-container; GPU only if video pairs are re-cut) and the H3 audio head-aligned re-sample (`--frame-head`/`--latent-head` measured from the probe's `owner_per_frame`). Then on CPU: re-run the ladder (`c15_real_v4`) with the ACE refit-D-on-E(raw) experiment and the composed rank-64 end-to-end map; C16 v3 with clip-grouped holdout; H3-0 v6 with three seeds (this one is GPU0, 600 steps/arm ≈ 45 min/seed on the 3.19M trunk [assumed from 2,614 s/4,000 steps]). | see 1.4 | `c15_real_v4`, `c16_L12_v3`, `h3_0_v6_seed{1,2,3}` receipts; **Cyril R3** media rows |
| G0-10 | +36h → +48h | **Wave A round 2 on the 4B, informed by R2**: the excision ladder completed (remaining single layers, then pairs ordered by block distance), C2 at 8x on the sites R2 kept, C5+C2 and C7+C2 pairs (two-axis children through two `qf_build` passes: child of a child, lineage recorded), C6 at `--nope-keep 24`. Same review loop as G0-4. | as G0-3/G0-4 | second child index; frontier table |
| G0-11 | +48h → +60h | **Wave A on Qwen3.5-0.8B** for the fold-vs-random-init ladder prerequisite (D-R5-4): the same axes at the same ratios so stage D's control arm has a cheap parent; and the 0.8B GGUF round trip through the import lane's writer for a folded child (first Chimera-native export of a fold). | as above with `--parent /home/cereal/models/Qwen3.5-0.8B` | 0.8B child index; one Chimera-imported folded child with an exact-import receipt |
| G0-12 | +60h → +72h | **tps_probe cells 1–2** (write `distill/tps_probe.py` first, section 4.2): 0.8B full-param bf16 fwd+bwd at L=1,024 and 4,096 (+ 4B teacher inline); 4B LoRA r=64; 4B full-param with CPU-offloaded AdamW (hand-rolled). One JSON receipt per cell. These numbers gate every stage-D budget. | `$PY /home/cereal/omni-campaign/distill/tps_probe.py --cell ...` | `distill/tps/*.json` for **Cyril R4** |

### 2.2 P40 queue (GPU1 in flight; GPU2/3/5 free) — inference/eval only, Pascal venv for torch

| Slot | Window | GPU | Job | Done when |
|---|---|---|---|---|
| P-0 | continuous | GPU1 | The 27B q4_0-KV 256k sweep `oc-env-27b-gpu1-q4kv` (pre-fix module; started by the export lane). Poll its `.partial.json`; do not restart. Expected 262,144 point ≈ 60–65 min prefill [verified Phase-0 63 min]. | receipt `envelope-OC-Q36-27B-Q4KM-gpu1-q4kv.json` (count `points`, ignore `partial`) |
| P-1 | T0+2h | GPU3 | Re-run `kv_fit_search.py` q8_0 on the P40 with the shipped tool so the receipt carries `load_timeout_s`/`wait_exceeded_load_timeout`/`peak_by_attempt_mib` (8 launches, ~35 min). | `kvfit-…-q8kv-p40gpu3-v2.json` |
| P-2 | T0+3h | GPU5 | Serve each Wave A child's Q4_K_M GGUF (from G0-5) on a P40 for parity at 4,096 ctx (short-context sanity that the dead `p40probe` container used to provide) and one 32,768 needle point. | parity + envelope receipts per child on sm61 |
| P-3 | after P-0 finishes | GPU1 | `kv_fit_search.py --kv q4_0` on the 27B (how far past 262,144 q4_0 KV actually goes on 24 GB) and the same on GPU0-class later. | fit receipt |
| P-4 | T0+4h | CPU (44 jobs) | Build `llama-imatrix` into `llamacpp-imatrix/` (separate root; `llamacpp/bin` untouched while P-0 runs). Then IQ4_XS-with-imatrix for the 0.8B and re-parity on GPU5. | build receipt with llama-imatrix sha; `parity-Q35-08B-IQ4XS-IMATRIX.json` |
| P-5 | T0+20h | GPU2 | lm_eval bench rows for Wave A children in fp16 (P40 has no bf16 tensor cores) as the second bench surface, labelled. | `bench/` rows |

### 2.3 GPU4 (3090, Codex) and GPU6 (RTX 8000, 46 GB) — only after the relay claim is recorded (2.0 step 0.2)

| Slot | GPU | Job | Notes |
|---|---|---|---|
| G6-1 | GPU6 | Take over G0-7/G0-8 (K3 streamed layers + C14): packed experts 15.7 GB + body + activations fit; fp16 compute after a 100-step overflow check (sm75, no bf16 tensor cores). Frees GPU0 for Wave A. | The import lane already ran R-Q/R-K3 on GPU6 without incident |
| G6-2 | GPU6 | H3 DiT block capture for C16 (bf16 40.2 GB pruned `minimax_h3_fl2va_pruned_bf16`, our own `docker run` from the r5 image, `-v /home/cereal/omni-campaign:/work`): one block's I/O on fixture latents; the H3 DiTBlock coreset 14336→3584. Also H3 denoiser-distillation pairs (per-σ x0 predictions on VAE-encoded curated clips) — requires the D12 catalog flip (section 5). | Never restart `comfyui_h3_gpu4`; it is gone anyway |
| G6-3 | GPU6 | 200k long-context lane for the 4B parent in bf16 HF (46 GB fits ~258,800 tokens? [assumed — the 24 GB card OOMs at that length; at 2x VRAM it is plausible, not measured]) so bank v2's lc-200k rows get a parent control without llama-server. | if it OOMs, the llama-server runtime is the control (R3 note) |
| G4-1 | GPU4 | Second Qwen queue: Wave A reviews (G0-4 loop) in parallel with builds on GPU0; halves the 72-h critical path. | `--gpu-uuid GPU-2ce6a95c-…`, `--require-compute-architecture sm86` |
| G4-2 | GPU4 | tps_probe cell 3: 27B-U LoRA r=64 base split GPU6+GPU4 (`device_map` sequential, fp16 half on GPU6 with loss scaling, L=4,096, micro 1) — THE stage-D number (D-R5-1). Needs both cards, ≈1 h. | receipts `distill/tps/27b-lora-split.json` |

### 2.4 What Cyril reviews, and when

| Review | ≈ When | Package | What he rules on |
|---|---|---|---|
| **R1 — parent surfaces** | T0+4h (G0-2 done) | `calibration/out/SCORING-SHEET-Q35-4B*.md`, `SCORING-SHEET-B2-Q35-4B.md`, `SCORING-SHEET-LC{8k,32k,64k,200k}-Q35-4B.md`; READ files with verbatim parent responses on both surfaces | (a) score the 4B parent per case on the `--thinking off` surface (this becomes `PARENT_REVIEW_BAND[Q35-4B]` and `bands.json`); (b) which surface is THE review surface (template-default = 13 cap-rides; thinking-off = 16/16 complete) [proposed: thinking-off primary, template-default published]; (c) bank v2 draft authorship: keep/replace lanes, re-cut the long-context ids to real token counts (moves the draft digest), vision/AV composition; (d) copy-measure retune on reasoning surfaces (0.60 union coverage reads think blocks as ECHO) [proposed: unchanged, published as caution]; (e) PARENT_BAND_TOLERANCE 2 carry vs two-reader re-measure; DEGRADED_BAND = ceil(parent/2) |
| **R2 — Wave A child index** | T0+16h (G0-4 done) | `$OC_ROOT/oc01/human-review/<id>/REVIEW_<TAG>.md` ×15, `SCORECARD.md`, `REVIEW_QUEUE.md`, `oc01-child-index.json`, GGUF/parity/envelope receipts for the top 3 | (a) per-case readings for each child on bank v1 + the five bank v2 lanes (the canonical three-box column is his read, the machine's is an upper bound); (b) which classes stay OPEN and which rungs deepen (8x, pairs); (c) "partially capable" = ≥1 canonical signal lane on any surface [proposed] → KEEP/DROP list for retention (`delete` will refuse anything he marks partially capable); (d) column B (copy-backs refused) as default with A published [proposed] |
| **R3 — K3 + media + export** | T0+36h | `k3/k3_layer1_exactness_receipt.json` + the A_log verdict note; `folds/L01/K{17408,8704}` receipts; `media/out/{c15_real_v4,c16_L12_v3,h3_0_v6*}` + decoded WAV/MP4 artifacts beside parent artifacts; `export/receipts/envelope/*gpu1*` | (a) **D-a2b** A_log axis reading (trim-to-96 vs head_dim-128 broadcast) — the whole K3 lane and the D4 `decay_log_scale` mapping depend on it; (b) D-a1 calibration T and cache set, D-b1 rung policy, D-a3 staging permission; (c) D-d2 absorption line and the ACE reading (under-fit vs D5b companion decoder), D-H3-geom, D-d1 (3D last); (d) blind 1–5 on any media-out artifacts; (e) D12 catalog flip for H3 (evidence_uri + deployment_config_sha256) and the H3 denoiser-distillation arm (D-R5-7); (f) which of q8_0-at-233k / q4_0-at-262k is the export target and whether P40 throughput (3 tok/s at depth) is acceptable |
| **R4 — stage-D go/no-go** | T0+72h | `distill/tps/*.json`, the omni pre-registration draft (`distill/PREREGISTRATION_OMNI_STAGE_D_v1.md`, section 4.2), clone receipts for the children he kept, the registry walk + token census | D-R5-1…D-R5-11: 27B LoRA tok/s floor (100) → graft + last-N LoRA; Q8 GGUF teacher for SAMPLING (yes/no); on-policy fraction (0.25 after 20M); control arm ≤4B only; distill in qwen3_5 layout then exact-import (D-R5-5); K3 KD token budget; 4B CPU-offload full-param vs LoRA; stage-T mix 3:1, ≤2 epochs, KL weight; rung tokens 10M/50M/200M and lr re-registration; disk: ≤2 27B lineages in flight |

---

## 3. Checkpoint / clone / delete policy (authoritative text: `OC/CHECKPOINT_POLICY.md`, enforced by `OC/ledger/oc_checkpoint.py`)

Cyril's ninth ruling (2026-09-01): checkpoint and clone before training on top so nothing regresses; delete bad checkpoints; let him review anything at least partially capable. The policy turns that into rules the registry refuses to break.

- **R0 What a checkpoint is.** A directory of sharded safetensors + config + tokenizer files, a build receipt, and a sealed registry record. Parameter counts are READ FROM THE SAFETENSORS HEADERS; a typed count that disagrees is refused; a checkpoint with nothing to read needs `--allow-asserted-parameter-counts` and is labelled `asserted-by-operator` in the record, the row, the scorecard's `ratio from` column and the package. `net_ratio` is computed, never accepted. `verify` (cheap) re-reads both counts on every run; `verify --deep` re-hashes bytes.
- **R1 Naming.** `OC<NN>-<F|D|T>-<PARENTCODE>-<TAG>` (TAG ≤16 A-Z0-9), parents `PARENT-<CODE>`, clones `<source>.clone-pre-<D|T><k>`. Parent codes: Q35-0.8B, Q35-4B, Q35-9B, Q38-27BU (source), Q38-27B (control), K3-L<i>, DK3-<rung>, H3, ACE, QIMG, HY3D, QVL32.
- **R2 Clone before ANY training step. No exceptions.** Source must be reviewed (or `--allow-unreviewed-source`, recorded); source directory must still digest to its sealed value; byte copy (hardlink only behind `--mode hardlink --allow-hardlink` and documented as NOT a valid pre-training clone); digest asserted equal; sealed `clone-receipt.json`; `chmod -R a-w`; status `cloned`; never deleted while any descendant exists. A distill/finetune child whose `--parent-checkpoint-id` is not a clone is REFUSED at register. Training writes to a NEW directory only.
- **R2b A fold must name what it folded.** `--component name:parent_params:child_params[:tensor_prefix]` per folded component; at stage fold each ratio ≤ 0.25 (≥4x) is enforced; with a tensor prefix the bytes decide the counts. `quantization: none` is a literal at fold and distill.
- **R3 Checkpoint each thing.** Every stage ends in a sealed child + dossier + review before the next stage starts; no stage starts from an unsealed checkpoint; Q4 exactly once at the very end, re-graded as its own `.Q4` row.
- **R4 Delete bad checkpoints — the registry is the only path.** `oc_checkpoint.py delete` prints fourteen clauses PASS/REFUSE and does nothing without `--confirm`: (i-a) status is `bad`; (i-b) a review exists; (i-c) canonical signal = 0 on every surface; (i-d) no media artifact awaiting Cyril's score; (i-e) at least one measured CONTENT miss (an artifact-only child can never be deleted); (ii) not a floor/frontier holder on any column; (iii-a/b/c) an exact rebuild command, an existing retained source, a ≥60-char argued reason; (iv) not a parent; (iv-b) no descendant (clones included); (iv-c) not under a protected prefix (`/mnt`, `/home/cereal/models`); (v) a review record naming a package whose MANIFEST.json verifies; (vi) the directory still digests to its record. The chain lock is held for the whole command and the delete record is built before the unlink. Retention: KEEP/DROP lists hand-written from the wave's verdicts; once per wave rebuild one DROPped child from its receipt and compare digests.
- **R5 "At least partially capable" → package, never delete.** [proposed] ≥1 canonical signal lane on ANY surface (bank v1, bank v2, or media-out with non-zero content — a WAV with `audio_rms <= 0` may not claim `non_empty`). Status `partially_capable` is COMPUTED from the review counts; it lands in `REVIEW_QUEUE.md`.
- **R6 Never delete:** a parent, a provenance-verified local parent copy, a clone, a GGUF named in a ledger row, anything on `/mnt/ai_data`, anything under `/home/cereal/models`. The registry enforces the first three and the two path prefixes; GGUFs named in ledger rows remain an operator instruction.
- **R7 The review package** (`oc_package.py build`): README, `REVIEW_<TAG>.md` in seven sections (what was folded/trained with per-component ratios; benchmarks vs parent and vs pre-training checkpoint; the frozen review in full with child and parent side by side verbatim and a per-case table carrying the sealed grade AND the canonical three-box reading; bank v2 in full; media artifacts with diagnostics and parent artifact beside; export path + digest + exact llama-server command; reviewer notes and what is NOT known), READ-/READ-PARENT-/READ-B2- files, record chain, all receipts, MANIFEST.json with `package_sha256` re-verified on write. Cyril's scoring is written back as `cyril-score-<TAG>.json` and sealed as a new record (`-cyril` instance).
- **R8 Statuses** {candidate, partially_capable, unmeasured, bad, cloned}; computed by `status_for` from counts; `--status` may only assert agreement; only `--cyril-ruling` overrides. `lanes_total` < 2 is refused (a 1-lane review cannot label a child `bad`).
- **R9 The ledger.** `$OC_ROOT/ledger.jsonl`, append-only, canonical bytes, single-writer O_EXCL lock (local disk only — never on CIFS), two schemas (ancestor `neural-forge-lfm2-matrix-cell-ledger-v1` passed through; `neural-forge-omni-checkpoint-ledger-v1` with `campaign="omni-2026-09"` and 38+ columns incl. `parameter_count_source`). `merge` byte-concatenates shards, appends under the lock, refuses unless the old bytes are a prefix, fsyncs. Every mirror into `OC/` is byte-compared to the server authority; a line that is not canonical bytes is refused by `verify`.
- **R10 Dashboards** `SCORECARD.md` (ordered by canonical signal per unit of ratio; unreviewed rows sort last) and `REVIEW_QUEUE.md` are regenerated from the registry, never typed.
- **Disk arithmetic for 27B lineages**: child + clone + stage output = 3 × 55 GB = 165 GB per lineage step → at most two 27B lineages in flight on 537 GB (D-R5-11). Rung checkpoints at 27B are 55 GB each; keep rung + last safety; optimizer state only on the last rung.
- **Open policy decisions for Cyril** (`CHECKPOINT_POLICY.md` last section): the partially-capable floor (1 lane); whether `--allow-unreviewed-source` and hardlink mode should exist at all; a fifth status `parent`; whether the two `state_sha256` conventions (safetensors_io vs proof.hash_state_dict) stay unequal.

---

## 4. The stage ladder: fold → distill → finetune, with the results package for each

Cyril's order (2026-09-01): fold every named source into ONE dense trunk; distill from the parents for all modalities; finetune on the curated set; checkpoint + clone before each training step; delete bad checkpoints; package partially-capable ones. Stage order is fixed; each stage ends in a package he reviews before the next starts.

### 4.1 Stage F — FOLD (closed-form, no gradients, ≥4x per component, storage in parameters)

Ladder inside the stage: Qwen3.5-0.8B → 4B (this window) → 9B → Qwen3.8-27B-Uncensored (needs the StreamedHarness: one decoder layer on GPU, residual `[T,5120]` bf16 cached on CPU; Gram-only solves — `capture_grams` exists, untested). Fold classes and their state:

| Class | Tool | State | Next rung |
|---|---|---|---|
| C1 vocab coreset | qf_vocab | fixture 1.33x only | 4B `--vocab-keep 62080` (G0-3) |
| C2 MLP channel coreset | qf_mlp | fixture 4.00x exact | 4B 9216→2304 (4x), →1152 (8x) |
| C3/C4 excision | qf_excise | fixture, causal assertion proven | 4B singles → pairs by block distance; each full layer −6.25% KV at 27B |
| C5 GDN value-head (count / dim) | qf_gdn | fixture; beta refit fixed | 4B 32/16→16/8; dim 128→32 |
| C6 NoPE-dim contraction | qf_attn | fixture; per-head optimum proven | 4B `--nope-keep 48` (head_dim 112) then 24 |
| C7 kv-head merge (kv-only / full-group) | qf_attn | fixture; kv-only gate exact | 4B 4→2 both arms; 27B 4→3, 4→2 |
| C8 residual width | — | NOT IMPLEMENTED | port `width_axis` per PORT_SPEC §2.7 |
| C9 terminal gain | qf_termgain | identity gate exact | 4B final norm (+ per-layer sites untested) |
| C10 MTP carry-through | qf_export | synthetic only | real 27B `model-mtp.safetensors` |
| C11 vision merger / C12 media graft | — | import lane skips `model.visual.*` (100.6M) and `mtp.*` (20.5M) | after the 27B import receipt |
| C13 Chimera-only blocks (AttnRes identity-init, SiTU, per-dim KDA decay, MLA latent) | import lane switches | S1 landed; identity-AttnRes is not a plain residual (ruling needed) | measured folds on a Qwen-imported trunk, never bundled trained deltas |
| C14 K3 expert-union coreset | c14_union_coreset | synthetic only; mechanics verified | real layer 1 at K=17,408 / 8,704 (G0-7) |
| C15 VAE absorption | mediafold c15 | real VAEs; ACE under-fit, H3 audio needs companion decoder, video plausible | refit-D-on-E(raw), ctx2 ACE, head-aligned H3 audio; Hunyuan3D/Qwen-Image adapters |
| C16 DiT block transplant | mediafold c16 | ACE block 12 at 4.00x MLP / 1.82x block | kv-head merge to reach ≥4x on the block; clip-grouped holdout; H3 DiTBlock on GPU6 |
| C17 chart 5376→hidden / C18 cross-tokenizer bridge | c16 charts / importing.bridge | charts rated noise (index-aligned corpora); bridge aligned 43/43 on fixtures | one shared corpus for host+parent rows; full-vocab `token_alignment` |

Wave sequencing [proposed]: Wave A = single-class children on 4B (this window); Wave B = pairs and the 0.8B ladder mirror; Wave C = 9B; Wave D = 27B-U via the StreamedHarness; Wave E = the composed Chimera Omni trunk (Qwen-27B-U import + K3 dense-layer grafts + media bands) — the object stage D distils.

**Results package per fold child** (R7): `REVIEW_<TAG>.md` §1 component table with every folded component ≤0.25x and every `*_frame_mismatch` named, §2 bench row vs parent, §3 bank v1 in full both surfaces, §4 bank v2 text lanes, §5 media (none at stage F unless a media class), §6 HF path + directory digest + GGUF path + digest + exact `llama-server` command + envelope point, §7 notes. Wave-level: child index, frontier table, `SCORECARD.md`, `REVIEW_QUEUE.md`, retention manifest (`--plan` then `--apply` after package), ledger merge report + ledger sha256 in the wave note.

### 4.2 Stage D — DISTILL the folded trunk from the parents

Inputs required before step 0 (section 12 of the R5 digest): (1) the sealed, packaged fold child (ARCH/checkpoint row + dossier + Cyril's reading) and its `.clone-pre-D1`; (2) tok/s receipts for every mode used (`distill/tps_probe.py` — NOT YET WRITTEN; cells: 0.8B full bf16 L=1k/4k; 4B LoRA r=64; 4B full-param CPU-offload Adam; 27B-U LoRA split GPU6+GPU4 fp16-on-Turing; 27B teacher-forced forward-only; llama-server Q8/Q4 generation tok/s; Chimera-native fla-KDA path vs reference loop; K3 streamed layer wall time; media teachers seconds/sample); (3) directory digests of every local parent; (4) registry integrity + token census (2.0 step 0.5); (5) corpus manifests with the contamination screen; (6) teacher-cache manifests with teacher digests; (7) the omni pre-registration document (`distill/PREREGISTRATION_OMNI_STAGE_D_v1.md`, sha pinned into every receipt: objective schedule, optimizer, rungs, probe surface, matched-compute definition); (8) the import-lane exact-embed receipt for any Chimera-native student plus a fla-KDA + SDPA-MLA training-path receipt; (9) the fp16 overflow check on GPU6 for the first 100 steps; (10) the H3 catalog flag flipped (D12) before any H3 output enters a provenance-gated path.

Teachers and how they run here: Qwen3.8-27B-U bf16/fp16 split GPU6+GPU4 (70 GB) for top-k logits PRECOMPUTED once per corpus (top-64 bf16 + int32 id + full-row logsumexp = 384 B/token → 200M tokens = 77 GB local) or layer-streamed on one 3090 (the fold lane's StreamedHarness doubles as the teacher engine); llama-server Q8_0 GGUF on GPU6 for SAMPLING the self-gen corpus (D-R5-2, [proposed] yes for sampling, no for logits); K3 teacher-forced top-k via the streamed per-layer pass at ~1M tokens/pass on the highest-value sequences only, cross-tokenizer scored on common boundaries (C18); H3 denoiser distillation (one DiT forward per (latent, σ), ~1/20 the cost of sampling) inside our own container on GPU6 after D12; ACE LM (64k audio codes) as a plain HF causal-LM teacher for the music band; Qwen-Image / Hunyuan3D per-σ predictions; speech teachers absent until Qwen3-ASR/TTS 0.6B is downloaded (D8).

Objective [proposed, mirrors the sealed LFM2 design]: text = top-k forward-KL (+ tail bucket) with LM-CE floor; K3 top-k KL on its subset (weight D-R5-6); on-policy GKD phase after 20M off-policy tokens at fraction 0.25 (D-R5-3); media = x0 regression to the parents' per-σ denoiser predictions on VAE-encoded curated media + the native x0 flow loss; music tokens = KL to the ACE LM; explicit anti-repetition (unlikelihood on ≥3-gram repeats) + EOS supervision (loops/echo/missing-EOS are a training objective; 20 decode settings × 300 lanes recovered 0 on LFM2). Optimizer [proposed re-registration]: AdamW (0.9, 0.95), wd 0.1, clip 1.0, bf16 autocast (fp16 + dynamic loss scale on GPU6), fp32 master for every trainable tensor; lr full-param fold-init 1e-5–2e-5, LoRA 1e-4–2e-4, media graft 2e-3→3e-4; warmup 100; batch 16×4,096 (≤4B) or 8×4,096 with accumulation (27B); rungs 10M / 50M / 200M tokens sealed inside one run; rolling safety checkpoint every 30 min keeping 2. Random-init control at byte-identical topology only on the ≤4B ladder (D-R5-4). Runtime: distil in the qwen3_5 layout (transformers + fla) then exact-import into Chimera (D-R5-5) until the import lane ships a fla-backed KDA + SDPA gated-MLA training path; Chimera-only blocks enter as measured folds (C13). Launch as `systemd-run --user --unit=oc-distill-<tag> --property=Restart=on-failure --setenv=CUDA_VISIBLE_DEVICES=<UUID>`; resume by missing rung; heartbeat `progress.json` with measured tok/s.

**Results package per stage-D child** (R7 + training additions): everything in 4.1 plus READ-<FOLDPARENT-TAG>.txt so parent → fold child → distilled sit side by side; `training-receipt.json` (init digest, teacher digests + corpus manifest shas + shard consumption in read order, objective schedule, optimizer, rung records with tokens/steps/backward calls/wall/gpu-hours MEASURED, loss per branch reported SEPARATELY (forward-KL unbounded vs JSD ≤ ln 2), free-run match/lcp series on a sealed 64-prompt probe every 1M tokens, NLL diagnostic-only, device inventory, seeds, pre-registration sha); `clone-receipt.json`; `lora-adapters.safetensors` beside the MERGED bf16 child when LoRA; §6 training curves; the ≤4B "fold-init vs random-init at matched tokens" ladder result with both views (initialization value; end-to-end GPU-h incl. fold cost); trainable-parameter table.

### 4.3 Stage T — FINETUNE on the curated high-value set

Input = the sealed, packaged stage-D child (`.D1` row) + `.clone-pre-T1`. Data = the master registry rows rendered through the Qwen chat template into `SequenceSpan` roles (instruction→user ignored; response→assistant supervised; tool traces→tool_call supervised / tool_observation ignored), labels via `build_label_mask` + `apply_ignore_index(-100)`, packed to 4,096 with segment ids (Qwen3.5's GDN mask path is positional by config — excised children must carry re-derived `layer_types`); media rows DECODED from sidecars (PNG/JPG→`image_to_native_patches` patch 32 → 3072; MP4→`video_to_native_patches` tubelet 2 → 6,144 so `native_feature_dim` must rise to 6,144 [import lane: TRUNK_FEATURE_DIM=6144 in profiles]; WAV→`waveform_to_native_segments` at 32 kHz; OCR as image; 3D via the new `volume` type once the D10 anchor pairing exists); sidecars verified with `timeout 15 stat`, missing → row dropped and counted; the curated : distill-sample mix 3:1 [proposed D-R5-9], ≤2 epochs (token census first), contamination screen on every row and every teacher sample. Objective = LM-CE on supervised spans + native x0 flow on decoded media targets + optional KL-to-teacher regularizer 0.1–0.3 against the stage-D cache; MTP/aux (flow/grounding/sync) losses are UNWIRED in Chimera today and are not part of stage T until wired. Optimizer one decade below stage D (full-param 5e-6–1e-5; LoRA 5e-5–1e-4), warmup 50, cosine, batch 8×4,096; checkpoint every 25M tokens and at epoch ends; eval every 10M tokens on the 28-row registry eval split + bank v1 (never trained on) + bank v2. Long-context lanes from the BUILT ≥32k corpus at 16k–32k only if the activation budget allows (27B LoRA at 32k ≈ 22 GB activations → micro 1 on the split, or last-N-layer LoRA); 256k behaviour is inherited from the parents and MEASURED on the Q4 export, not trained.

**Results package per stage-T child**: as 4.2 plus the registry eval split scores and a per-route table; §5 media artifacts (child, pre-training checkpoint, parent side by side, wall-clock per clip vs LTX 331–361 s and H3 turbo); the Q4 export ONLY at the very end as its own `.Q4` row graded on every surface (bank v1/v2, canonical lens, Cyril's blind media 1–5, bench row, 256k needle). "100% up to snuff" is read on those surfaces; every line is a proposed decision line.

---

## 5. OPEN ITEMS

### 5.1 Unverified claims (things this book relies on that no receipt yet proves)

1. **Every fold-quality number in qwen-fold is on a random-init fixture.** The value-side repair path (out_proj re-solve, whitened LRA, decay/beta refit) has never reduced error on a parent with structure. G0-1/G0-3 produce the first real numbers.
2. **K3 A_log axis (D-a2b).** Disk F32[128]; code says [96]; import lane trims 128→96 with layer-0 exactness 3.47e-8 (evidence for the trim); k3 harness resolves by shape and uses `use_gate_in_kernel=False`. The two lanes have not been reconciled on real weights of a KDA layer with non-trivial decay; G0-6 decides.
3. **No real K3 layer has run through the k3 harness** (headers only). Phase-0's layer-1 probe (5.3 MB/s read, 245 s dequant, MoE forward 2.48 s CPU) is the only real-layer measurement.
4. **Share read rate when idle** (5.3 MB/s contended, 25.5 MB/s clean measured once) sets the K3 calibration price (16 h vs 3+ days per 93-layer pass); not re-measured with the downloads finished.
5. **`Qwen3_5ForCausalLM.from_pretrained` on the on-disk `model.language_model.*` keys** is assumed NOT to load; all lanes use `Qwen3_5ForConditionalGeneration`. Confirmed only indirectly.
6. **Qwen3.8-27B-Uncensored under transformers 5.12.1** (saved by 5.15.0; `output_gate_type: swish`, no `attn_output_gate` key): loads? q_proj shape? Never loaded here. Spec Q1 blocks the 27B import profile.
7. **GGUF conversion of folded children** (non-uniform `intermediate_size`, changed `layer_types`, GDN head counts/dims, head_dim 80/96/112/192 with partial RoPE 64, MTP skipping) — convert_hf_to_gguf behaviour is assumed; G0-5 is the first test. llama.cpp kimi-k3 loader accepting an all-dense file (`expert_count 0`) is assumed (Q8 of the import spec).
8. **lm_eval on qwen3_5 checkpoints** in `/home/cereal/ai-benchmark-lab/envs/lm-eval` — untested; the parent reference rows exist only for LFM2.
9. **tok/s for every training mode** — all assumptions (≈180 tok/s 27B LoRA split, ≈550 9B LoRA, ≈4k 0.8B full). The LFM2 design's generation estimate was 8.7x optimistic. Nothing >230M has ever been trained on this box.
10. **FSDP over mixed sm86+sm75 without NVLink** and **fp16 overflow of Qwen's GDN path on the RTX 8000** — unmeasured.
11. **GPU6 200k bf16 HF prefill fits in 46 GB** — plausible, unmeasured (G6-3).
12. **Registry integrity after the 08-19/08-22 cleanups** (no per-shard sha walk yet) and **per-route token count** (never censused). Per-entry license fields (which v9_base video/image rows are LTX rollouts) uninspected.
13. **The P40 27B sweep** in flight runs the pre-fix `envelope_bench`; if its 262k point is refused it dies with KeyError; read `points`, not `partial`.
14. **calibration `--thinking off` dossier** carries v1 kernel semantics; the r5→r6 two-surface table is at least two variables apart (G0-2 closes it).
15. **Media: ACE under-fit cause** (D fitted on true latents, not E(raw)); **H3 audio head ownership non-identity**; **H3-0 separation on one seed**; **video verdict uninformative** (parent round-trip 9.91 dB on the procedural corpus).
16. **Export `fa-negative-control` R3 premise** rests on an observation made before the join was hardened (same binary hash, so it joins; not re-run).
17. **Phase-0 claim "all seven GPUs are allocated to the campaign"** vs the plan's D14 "claim GPU4/GPU6 via relay" vs memory "gpu-slot-6 GRANTED" — the relay thread is the authority; record it (2.0 step 0.2).

### 5.2 Missing tools (must be written before the step that needs them)

| Tool | Needed by | Spec source |
|---|---|---|
| `qf_evidence_build.py` — sealed evidence manifest from the registry (+ ≥32k-token sequences for the long-context lanes, which must be BUILT) | Wave A capture (G0-1) | `qf_corpus.py` schema; PORT_SPEC §2.1; R5 §6.4–6.5 contamination screen |
| `seal_omni_architecture_child.py` + `run_omni_frozen_review.py` (family-neutral loaders; directory digest instead of single-file; bf16 review dtype; new receipt schema `neural-forge-omni-arch-cell-receipt-v1`, schema v4 for pinned-remote-code Chimera children) | first omni ARCH row in the ledger; qf_export frozen-review hook | R3 spec §5.2, C(1)–(8) |
| `bands.json` per parent + calibration sentinel in the seal queue | any sealed verdict on a child | R3 §D, P4–P7; needs Cyril's R1 scores |
| C8 `width_axis` port (basis writer/reader, `fold_reader`/`fold_writer`, tied-head gain solve) | Wave B | PORT_SPEC §2.7 |
| `StreamedHarness` (27B) + Gram-only C2/C5/C6/C7 paths exercised | Wave D, 27B teacher-forced logits | PORT_SPEC §3.8; qf_harness `capture_grams` |
| K3 tokenizer wiring (`encoding_k3.py`/`tokenization_kimi.py`; `tiktoken`/`blobfile` presence unchecked) + curated-shard corpus for the K3 calibration pass | K3-1 on real text (currently random ids) | R6 a.5; import lane's `TikTokenBpeTokenizer` can be reused |
| `tps_probe.py` (six cells) | stage-D budgets (G0-12, G4-2) | R5 §4 |
| Omni training receipt module (copy of `stage2_training_receipt.py` with a new pre-registration sha) + `PREREGISTRATION_OMNI_STAGE_D_v1.md` + teacher-cache writer (`teacher_cache/<teacher>/<corpus_sha>/shard-*.{ids.u32,topk_ids.i32,topk_logits.bf16,lse.f32,manifest.json}`) + corpus generator port of `stage2_generate_teacher_corpus.py` (uint32 ids, Qwen teacher, 4,096-token rows) | stage D | R5 §1.1, §5.6, §6.5 |
| Chimera fla-backed KDA training path + SDPA gated-MLA + media sidecar decoder → `ChimeraBatch` adapter | Chimera-native stage D/T; stage T media rows | R5 §1.3, §6.2 |
| Hunyuan3D ShapeVAE / Qwen-Image VAE adapters + D10 voxel-SDF anchor pairing; kv-head merge (rung 2) for C16 | C15-3D, C15-image, C16 ≥4x block | R6 (d), (e) |
| `llama-imatrix` build in a separate root; `--runner docker` exercise; media-output GGUF (D5b) | IQ4_XS with imatrix; media export | export spec gaps 1, 7 |
| vision/AV decode path in `oc_run_frozen_bank` (processor, pixel batching, asset loader) | bank v2 vision lanes | calibration gap 3 |
| ledger: `--parent-directory` protected-prefix check; `verify` PROBLEM-not-raise on symlinked subdirs; `audio_rms` required from the media runner | before first 27B registration | ledger review-2 notes 1–3 |

### 5.3 Pending claims and rulings

**Claims (relay / operational):** GPU4 3090 (Codex-reserved) and GPU6 RTX 8000 (Codex-owned) — D14, confirm via relay before use (`agent-comms`); Codex D4 import-lane ownership clock (Claude's `claude/omni-import` branch is at S5; Codex has been asked to review nine items in `IMPORT_LANE_STATUS.md` §9); permission to delete OUR OWN staged K3 shard copies for rolling staging (D-a3; default: stream experts, stage body only); Qwen3-ASR/TTS 0.6B download (D8) for a speech teacher.

**Rulings for Cyril (all proposed decision lines, grouped by review):**
- R1: review surface (thinking-off primary); PARENT_REVIEW_BAND[Q35-4B] from his per-case read; PARENT_BAND_TOLERANCE 2 carry vs re-measure; DEGRADED_BAND = ceil(parent/2); bank v2 authorship (composition 15 text + 4 vision + 2 AV-in + media-out; re-cut long-context ids to real token counts; long-context parent control runtime — HF layer-streamed vs llama-server GGUF); copy-measure retune on reasoning surfaces (default unchanged); `logits_to_keep` deviation acceptance.
- R2: "partially capable" = ≥1 canonical signal lane on any surface; column B default with A published; the KEEP/DROP list; which classes deepen; 4B full-param via CPU-offload Adam vs LoRA (D-R5-8) — informs which children get cloned for stage D.
- R3: D-a2b A_log; D-a1 T=65,536 minimal cache; D-a2 exactness tolerance (rel-L2 ≤2e-2, top-16 ≥99%); D-a3; D-b1 rung policy (stage-1-only at 65,536); D-b2 form E scoring / form M export; D-c1 first R-full at K ≤32,768; D-K3-review (R-proxy is the only per-layer instrument); D-H3-geom (video token = tubelet 4 × patch 32, feature 12,288); D-d1 3D last; D-d2 absorption line (PSNR ≥ VAE round-trip −3 dB); D-e1 depth-fraction pairing; D-f1 H3-0 pass rule; D12 H3 catalog flip (`evidence_uri` https placeholder + `deployment_config_sha256` per provider) and D-R5-7 denoiser distillation; export target (q4_0 KV at 262k vs q8_0 at ≤233k) and P40 acceptability; `kda_qk_norm_kind` default flip (Q2); `attnres_init='identity'` semantics (S1 gap); spec Q1 `output_gate_type='swish'` reading for the 27B; ledger_band_logit_mask default (Q3); `volume` in REQUIRED_MODALITIES (Q6); SiTU export rung β=1e4 vs silu (Q5).
- R4: D-R5-1 (27B LoRA tok/s floor 100 → graft + last-N LoRA); D-R5-2 (Q8 GGUF teacher for sampling only); D-R5-3 (on-policy 0.25 after 20M); D-R5-4 (random-init control ≤4B only); D-R5-5 (distil in qwen3_5 layout, exact-import after); D-R5-6 (K3 KD budget/weight); D-R5-9 (stage-T mix 3:1, ≤2 epochs, KL 0.1–0.3); D-R5-10 (rungs 10M/50M/200M, lr values); D-R5-11 (≤2 27B lineages; media caches may spill to `/mnt/ai_data` for writes); QLoRA (default NO); registry rows' quality-tier weighting (primary/supplemental tiers exist only inside batch manifests); whether the 1,837 un-appended candidate rows (batches 000148–000165) are duplicates or lost media rows; whether Cyril's "high-value dataset" also means the manual_review_batches exports.

### 5.4 Where this book's outputs land

Server: `/home/cereal/omni-campaign/{qwen-fold,ledger,k3,media,export,calibration,distill,datasets}/`, `$OC_ROOT/ledger.jsonl`, `$OC_ROOT/registry/`, `$OC_ROOT/oc01/{children,human-review,shards,logs}/`, `$OC_ROOT/SCORECARD.md`, `$OC_ROOT/REVIEW_QUEUE.md`. Repo mirror: the same tree under `OC/`, byte-compared after every wave; `OC/OPEN_ITEMS.md` (to be created from section 5 and kept current); `OC/NOTES-OC-A.md` for the first wave's note with the ledger sha256.
