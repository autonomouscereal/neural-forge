# R13 Definitive Action Plan â€” Neural Forge Intelligence-Retention Failure

## 1. Unified causal story (committed)

**r12 collapsed because moderate per-boundary solve error, made large by contraction forced past the parent's measured functional width, compounded multiplicatively across 14 pretrained nonlinear boundaries, was amplified by 100% teacher-forced (off-policy) evidence, and terminated through the fleet's only approximate token interface â€” while an acceptance chain that measured nothing composed guaranteed the wreck would be published rather than caught.**

The causal chain, in order of leverage (all four lenses converge):

1. **Pretrained functional rank sets per-boundary error scale.** The chimera toy's quasi-linear parent yields per-tensor fits at 0.002â€“0.14 rRMSE; real pretrained parents start at 0.13â€“0.55 at layer 0. The decisive A/B is the cross-family control: identical geometry, random-weight parent folds 6 layers with flat error (~2e-02), pretrained parent walls at layer 3 with upstream error-energy ratio exploding 7.04e-16 â†’ 1.54e-01 in two layers.
2. **Depth compounds it.** 3 pretrained boundaries = partial signal (26â€“50%), 6 = below-zero assembly (logit RMSE 1.077 with all 14 per-branch guards passed), 14 = total collapse (terminal rRMSE 0.809, free-run token match 0.44% â‰ˆ chance). Per-layer acceptance constrains only sign(local delta); terminal error composes as sum of deltas times downstream amplification factors nothing measures.
3. **Contraction beyond functional width made the per-boundary deltas large.** r12's own capacity option bank predicted ~0.0965 defect/layer, collision fraction 0.875, ~70% unrepresentable activation tail at the forced 2560â†’321 (8x) MLP contraction. The toy took its 4x on residual width only, retaining 96.2% chart energy. Nobody has measured a real parent at 4x width; the 1.35x sparse-semantic children scored 10â€“12/16 useful, so the frontier between 1.35x and 4x is unmapped.
4. **Exposure bias amplified composition.** All fold evidence was teacher-forced parent-state rows (on-policy round_count = 0 at fold time); the child's own visited states â€” where residual cosine has fallen to 0.574 â€” had zero evidence. Teacher-top1 1.86% vs free-run 0.44% is this gap measured directly.
5. **The 16x approximate vocab bridge is where it surfaced.** D is the only run with an inexact interface and the only wipeout, with an interface-shaped signature (vocabulary-stratum drift). SmolVLM's internal contrast shows interface exactness alone is worth ~2x parent agreement (0.332 vs 0.634 logit RMSE).

The acceptance machinery was not the physics but was the enabler: layer acceptance reduced to "metrics finite" (diagnostic-only), trajectory gate to "finite + cosine > -1", whole-output gates never ran (ABI smoke evaluator), all numeric candidate gates disabled, and per-layer metrics were computed, hashed, and discarded â€” so the run could neither stop itself nor leave a reconstructable depth profile.

**Explicitly ruled out as causes:** chart quality (real-parent held-out retained energy 0.957â€“0.983, better worst-case than random), raw evidence-row count (r12 had >100x rows/neuron, richest of all four, worst outcome), evidence starvation above the ~2x floor, and moving-gauge solver complexity per se (the simple-ridge lane composes to worse-than-zero at depth 6 too).

**Confidence: ~85%** on compounding-of-per-boundary-error-through-pretrained-dynamics as the dominant mechanism (anchored by the random-vs-pretrained A/B). Moderate-high (~70%) on the specific ranking of the three amplifiers (contraction severity, exposure bias, interface approximation) â€” each has one clean internal contrast supporting it but they have never been varied independently on LFM2. R13's arm structure below is designed to separate them.

## 2. Top 8 changes for r13 (ranked by retention-impact x cost)

**Sequencing note:** items 5 and 8 run first chronologically (hours, not days); item 2 must land before items 4, 6, 7 can be judged.

**1. Consume on-policy evidence IN-SOLVE (configuration only â€” highest leverage per unit cost).**
- Site: `experiments/run_lfm2_dynamic_catalog.py` (`on_policy_refold_after_rounds` :1220, alternation driver :12703-12770); staged launcher `nf-launch-r12-round-refold.sh` (smoke-tested; a live r12 round is already running pinned to SOURCE_COMMIT 8369f42e â€” consume its output).
- Mechanism: base fold + N rounds x bridged refolds makes every closed-form solve minimize error on child-visited states, directly attacking the 1.86%-vs-0.44% exposure-bias gap. Keep fit prompts disjoint from manual review (verification exists).
- Success: receipt `on_policy_evidence.consumed_by_this_solve_pass: true`; free-run token match rises decisively above 0.44% (target: within 3x of teacher-top1 rather than 4x below it).

**2. Absolute composition-aware acceptance floor + advisory tripwire.**
- Site: `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py` â€” extend `validate_layer` (:9922) and `_validation_decision` (:6075) to also compute TERMINAL (last-layer `residual.post_layer`) held-out relative RMSE from rows `_heldout_layer_metrics` already loads.
- Mechanism: (a) fail-closed when terminal rRMSE reaches the no-branch baseline 1.0 â€” the exact bound the kernel lane already enforces (`hf_causal_kind_translation.py:1437`) and the LFM2 lane lacks entirely; (b) reject any layer whose install regresses the terminal metric beyond tolerance (depth-monotone non-regression); (c) loud advisory diagnostic (opt-in stop flag, default off) when counterfactual free-run match sits at chance. This is a safety bound like the kernel lane's, not an intelligence verdict â€” the manual-review authority policy stands, but the carrier lane's hard diagnostic-only requirement (`run_lfm2_dynamic_catalog.py:2073-2088`) needs a carve-out for this one terminal floor. Near-zero extra compute.
- Success: r13 receipt carries per-layer terminal rRMSE; a run headed for a wall recloses or aborts at the wall layer instead of burning 64 more hours and publishing; items 4/6/7 become judgeable on composed error instead of fit-row error.

**3. Exact-interface arm: decouple the token interface from the 4x body proof.**
- Site: launcher/bridge configuration; reuse the cross-family lane's strict one-row-to-one-row donor-subset vocab transport (65,536 â†’ exact subset, 0 uncovered rows) in place of the approximate 16x pair bridge for one r13 arm.
- Mechanism: isolates body-composition physics from the bridge; removes the component whose signature matches the collapse. Interface exactness measured worth ~2x agreement in SmolVLM. Do not re-couple the 16x bridged vocab until the multi-code readout gap is closed.
- Success: exact-interface arm shows no vocabulary-stratum drift in decoded output; the delta between the two arms quantifies the bridge's contribution to collapse.

**4. Cap per-layer contraction by measured functional width; bisect the 1.35xâ€“4x retention frontier.**
- Site: heterogeneous capacity option bank / global exact-budget topology solver config in `run_lfm2_dynamic_catalog.py`.
- Mechanism: refuse per-layer widths whose predicted post-activation tail fraction exceeds a floor (r12 knowingly installed layouts with ~70% unrepresentable tail and predicted defect 0.0965/layer, 14 times). Treat exact-4x as a directional target walked down from a defect-bounded layout; run one arm at an intermediate global compaction (e.g. ~2x) to map the frontier the 1.35x (10-12/16 useful) children left open.
- Success: r13 layouts all satisfy the tail-fraction floor; the intermediate-compaction arm scores nonzero on manual review, establishing where retention dies between 1.35x and 4x.

**5. Depth-ladder probe + cache-only first-divergence diagnosis BEFORE the full 14-layer run.**
- Site: existing fold machinery at depths 3/6/10; Codex diagnostics commits 60eb928b/9bc0d735 against the sealed r12 child's `cache/child-final` and `cache/child-reload` (514 files each) â€” zero refolding.
- Mechanism: locate LFM2's own wall the way the cross-family run located SmolLM2's at layer 3; localize the first layer where r12's child residuals cross vocab decision boundaries. Tells the reclosure and EIV work where to spend effort, and prevents committing ~2 GPU-days blind.
- Success: a measured wall depth for LFM2 under the r13 solver; a per-layer first-divergence profile for r12; go/no-go on the full-depth run conditioned on the wall moving.

**6. Reverse reclosure sweep against RECAPTURED states.**
- Site: `src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py` reverse pass (:3803-3876) â€” insert a fresh child recapture between forward and reverse sweeps (machinery exists: `_body_internal_recapture`, adapter :4961) and re-solve writers back-to-front against terminal-anchored targets measured on actual downstream response, replacing the frozen-writer-input identity-transport approximation. Bump the policy schema (`authentication_body` pins sweep counts at :782-783).
- Mechanism: the direct counter to measured error accumulation â€” each writer absorbs remaining terminal residual as it actually propagates through downstream nonlinearity, not under a first-order skip assumption. Cost ~1 extra sharded replay (~2.2h at r12's 7,777s).
- Success: post-reclosure terminal held-out rRMSE (item 2's metric) drops materially versus forward-only reclosure; per-layer amplification ratios in the receipt shrink back-to-front.

**7. Wire the NOTES-1G math pack into the layer fits (EIV ridge + margin row weights).**
- Sites: `hf_lfm2_dynamic_closed_loop_adapter.py:9781` MLP compile call â€” pass `input_error_covariance` (from `lfm2_estimate_input_error_covariance`, structural fold :924, estimated on the previous accepted layer's parent-vs-child paired rows) with the {0, 0.25, 1.0, 4.0} multiplier sweep and a multiplier=0 control arm; replace the uniform `1.0/total` profile in `_moving_gauge_row_manifest` (:4746-4763) with margin-derived `case_row_sample_weights` (floor/ceiling 0.25/4.0, terminal/lexical writer fits excluded) per NOTES-1G Â§1â€“2. Currently these have ZERO production call sites.
- Mechanism: EIV makes each solve explicitly robust to upstream child-state drift (the compounding term, inside the solve itself); margin weights point fixed fit capacity at the low-margin decision-boundary rows the child actually failed on. Expect fit-row RMSE to worsen while composed error improves â€” judgeable only via item 2.
- Success: multiplier>0 arms beat the 0-control on terminal rRMSE and free-run match; per-layer fit-row-weight profile receipts sealed.

**8. Receipts, conditioning, and gate-operating-point telemetry.**
- Sites: persist every per-layer `ValidationDecision`'s metrics and the trajectory gate's full `residual_layers` depth table (currently computed then discarded â€” r12's compounding signature is unreconstructable) into receipt/diagnostics; extend `Lfm2PriorCorrectionReport` (structural fold :1144) and `_anchored_ridge_replacement` (:346) with sigma_max/sigma_min/condition number/solver-path (solve vs silent gelsd fallback :1103-1122); pass `gate_operating_point_diagnostics=True` at the MLP compile site (params exist at structural fold :4543).
- Mechanism: r12's receipt contains zero improvement keys and zero condition numbers; stage-2 proved condition caps saturate on every real-model sub-solve (fitâ†’validation degradation 1.5â€“2.1x) and the 25%â†’6% gate-saturation collapse is invisible to RMSE. Cost ~1 GEMM/layer plus serialization. Adopt condition-capped ridge (cap ~1e4, `_condition_capped_solve` form) once r13's receipts show whether the body solves saturate.
- Success: r13's receipt alone suffices to reconstruct the full depth-degradation profile, conditioning regime, and gate-saturation drift of the run.

## 3. Efficiency config for r13

Pre-run engineering (all from the efficiency audit, all low-risk):
- **Land the uncommitted bridge-projection reuse diff** in `run_lfm2_dynamic_catalog.py` (+377/-31: `_bridge_projected_parent_caches` memo with sha256 reauthentication + per-record progress events). Run its two new tests plus `test_run_lfm2_dynamic_catalog_tokenizer_bridge.py`; coordinate the commit with the live on-policy round pinned to 8369f42e. (âˆ’4 to âˆ’7h; instruments the currently-silent 8.5h evaluator interiors.)
- **Collapse the install sweep** (`hf_lfm2_dynamic_closed_loop_adapter.py` ~:10860-11110) to per-step `replacement_weight_sha256` fingerprint authentication + ONE terminal capture compared to the preview's final sha256, behind `--dense-moving-gauge-install-verification fingerprint|full-recapture` (full recapture retained for certification runs). The sweep does zero new solving today. (âˆ’17.7h, largest single win.)
- **Deduplicate the fresh-reload evaluation** in `dynamic_fold_artifact.py` (:2822 vs :2978 evaluate byte-identical artifacts): bind the staged decision to `portable_child_state_sha256` and reuse on fingerprint match, fail-closed to full re-eval on mismatch. (âˆ’8.5h.)
- **Add a terminal restart capsule** (post-output-interface / pre-evaluation) â€” capsules currently stop at body layer 13 and moving_gauge_v3 refuses mid-body resume; a crash in the terminal phase currently costs the entire run.

Flag set (deltas from r12): `--dense-body-reclosure-mode moving_gauge_v3`; `--dense-closed-loop-replay-mode prefix_by_prefix_canonical`; `--dense-closed-loop-merge-adjacent-layer-captures` (proven: 13 merged captures); `--dense-replay-devices cuda:0,cuda:4` (hard cap 2 â€” P40s are sm_61-unsupported, RTX 8000 is a different class); restart capsules keep 2 + the new terminal capsule flag; the new install-verification flag; balanced-basis and global-topology phase caches ON; the new terminal-floor and telemetry flags from items 2 and 8.

**Wall-clock projection:** single-pass fold transaction ~44â€“48h (~1.9d) vs 95.2h measured for r12 (body ~18h + reclosure ~19.4h with install collapsed + final-norm/output-interface solves ~11.8h + evaluators ~11â€“14h + overhead ~1h). Each on-policy round + bridged refold adds roughly one body+reclosure block (~35â€“38h with the install collapse); base + 1 round â‰ˆ 80â€“86h â€” still under r12's wall-clock for two full passes. The depth-ladder probe and cache-only diagnostics (item 5) are hours and run first. The 11.8h of opaque final-norm/output-interface solve time is the next audit's target once item 8's telemetry decomposes it.

## 4. Explicitly do NOT do

- **Do not invest in chart machinery or more calibration rows.** Charts are measurably not the limit (real-parent held-out retained energy 0.957â€“0.983, better worst-case than random; the toy got 73% with one crude global chart) and r12 had >100x the stage-2 evidence-ratio floor. More teacher-forced rows attack a non-cause.
- **Do not treat per-layer/per-branch guard passage as evidence about the assembled model** â€” the L6 random control passed 14/14 guards and assembled to worse-than-zero. Only item 2's terminal metric speaks to composition.
- **Do not rewrite or abandon the moving-gauge solver lane.** The evidence points away from lane complexity as the cause (simple-ridge composes to failure at depth 6 too; the planner predicted the defect the run confirmed). Fix its acceptance and evidence, not its identity.
- **Do not re-run forced global exact-4x with the coupled 16x approximate vocab bridge unchanged.** That configuration is the one fully-explained data point we have.
- **Do not shard `build_lfm2_circuit_trace_cache`** (the W8C capture-shard item): the builds take 17 minutes each, not 17 hours â€” the brief's "17h silent serial reload capture" was a misdiagnosis of two 8.5h evaluator passes. Saves ~16 minutes; fold into a later debt pack. Mark the Wave-8 capture-shard stream landed-for-adapter-path-only (b799a1a7); NOTES-W8C.md was never written.
- **Do not pin `--balanced-basis-existing-phase-key-sha256` across any source change** (r11's fail-closed lesson).
- **Do not give the vocabulary-margin candidate gate verdict authority on the carrier lane yet.** Rehearse it on a non-carrier/toy exact-4x lane to calibrate thresholds, then run it record-only on r13 so the receipt carries per-layer decision-boundary-crossing evidence without violating the manual-review authority policy.
- **Do not port Procrustes frame transport.** The moving-gauge lane's shared `base_encoder` gives coherent frames by construction; the cross-width anchoring only matters if the lane ever moves to independently-solved per-boundary charts.
- **Do not attempt early-exit capture forwards in r13** (halving preview-sweep cost): it touches capture identity; schedule only after a toy dress rehearsal.
- **Do not skip the depth-ladder probe to save calendar time.** Committing another full 14-layer multi-day run before measuring whether the r13 solver moves LFM2's wall repeats r12's core process error: spending the full budget before the first composed measurement.

Key paths: `C:/Users/cereal/Documents/ai training/neural-forge/src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py`, `.../src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py`, `.../src/neural_forge/hf_lfm2_structural_residual_attention_fold.py`, `.../src/neural_forge/dynamic_fold_artifact.py`, `.../experiments/run_lfm2_dynamic_catalog.py`, `.../experiments/run_lfm2_residual_attention_fold.py`, `.../NOTES-1G.md`, `.../nf-launch-r12-round-refold.sh`, `.../reports/r12-certified-20260804/`, `.../reports/cross-family-real-20260804-pretrained/`; remote run at 192.168.50.222:/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12/.
