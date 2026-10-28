# NOTES-W13T — the ~2x ceiling arm: composed, derived, and refused at config time

Wave 13T, branch `codex/lrc-blank-compression`, local HEAD `9326a564`
(notes-only on top of `cbfd9ab9` — verified, `git diff --stat cbfd9ab9 9326a564`
is `NOTES-W13S.md | 685 +`, one file). Deployment
`neural-forge-postpub-e893767a` at SOURCE_COMMIT
`5739c7425a4ca90eda6ca2af35e7533bfec58fac`, tree `48386cb0…`,
`git status --porcelain` empty. No deployment bump was needed and none was made.

**Nothing was launched.** The composed ~2x arm is refused by the deployed
runner's own carrier ABI guards **at configuration time**, before any model
load, before any run directory exists. This is the `STOP and report` case the
task named, and the refusal is decomposed to three exact messages below with a
control that proves the rest of the composition is valid.

The budget and hidden-size derivation the task asked for was completed anyway,
against the deployed solvers, and it is the useful part of this wave: **the
exact 2x point is reachable — `114,846,592` counted, `110,978,816` dense body,
and H816 is one of 45 residual widths with an exact integer lattice there.**

---

## 1. What the ~2x arm is, and the one number that decides its shape

The ceiling map's rungs are counted-parameter ratios, but the physics that
kills retention is **body** contraction, and the two are very different here
because the token interface shrinks 16x on its own (65,536 donor rows → 4,096
target rows).

Parent `LFM2.5-230M`: `vocab_size 65536`, `hidden_size 1024`, `block_ff_dim
2560`, 14 layers, `tie_word_embeddings true`, `conv_L_cache 3`
(read from `/home/cereal/model-surgery-lab/models/LFM2.5-230M/config.json`).

```
parent unique parameters                          229,693,184   (inspect_model_circuits)
parent tied token table            65,536 x 1024 =  67,108,864
parent physical body            229,693,184 - 67,108,864 = 162,584,320
```

That last number is what the ladder should be read against:

| rung | counted total | counted ratio | child token interface | physical body | **body ratio** | manual review |
|---|---|---|---|---|---|---|
| A9 | 168,712,512 | 1.361447x | 1 x 4096x1024 tied + 1 route | 162,584,320 | **1.000x** | 3 useful / 7 partial of 16 |
| A11 | 170,646,400 | 1.346018x | 1 x 4096x1024 tied + 2 routes | 162,584,320 | **1.000x** | 3 useful / 9 partial of 16 |
| **~2x (this arm)** | **114,846,592** | **2.000000x** | 2 x 4096x816 untied + 2 routes | **104,294,144** | **1.559x** | **unmeasured** |
| arm A (4x) | 57,420,144 | 4.000220x | 2 x 4096x816 untied + 2 routes | 46,867,696 | **3.469x** | 0/15 |

The A9/A11 rows are *reconstructions* and they reproduce the sealed ratios
exactly, which is what validates the whole accounting model:

```
A9   162,584,320 + 4,194,304 + 1,933,888 = 168,712,512 ; 229,693,184/168,712,512 = 1.361447   <- docs/CURRENT_FOLDING_GOAL.md
A11  162,584,320 + 4,194,304 + 3,867,776 = 170,646,400 ; 229,693,184/170,646,400 = 1.346018   <- docs/CURRENT_FOLDING_GOAL.md
```

**The consequence for the ceiling map: "~2x counted" is a ~1.56x body.** The
frontier between 1.35x (body 1.000x, 10–12/16 useful) and 4x (body 3.469x,
0/15) is a *body* frontier, and the exact-2x counted point lands at 1.56x body —
much nearer the surviving rung than the counted ratio suggests. That is still
the right measurement to take; it just needs to be labelled by both numbers when
the sealed stage-1 report quotes it.

---

## 2. Budget derivation — the split, and why it is exact

### 2.1 How the 4x launcher's two numbers relate

`nf-launch-r12-round-refold.sh` passes `--counted-target-total-parameters
57420144` and `--counted-dense-body-parameters 53552368`
(= `--dense-heterogeneous-target-parameters`). The relation is a **fixed
subtraction, not a ratio**:

```
57,420,144 - 53,552,368 = 3,867,776 = 2 x 1,933,888
```

`1,933,888` is `LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE`
(`experiments/run_lfm2_dynamic_catalog.py:345`), the size of one A9/A11 route
channel, and the config validator enforces exactly that subtraction
(`:2278-2287`). The carrier publication's own receipt agrees:
`input_coefficient_count = 1933888`, `output_coefficient_count = 1933888`,
`continuous_semantic_coefficient_count = 3867776`
(`…/a9-a11-base-dual-route/4439a2b7-b301539b/complete.json`).

**The routes do not scale with the budget or with the hidden size.** They are
donor-side coefficient tensors over the 4,096 target *rows*; the packaging
receipt records `"source_target_table_packaged": false` and
`"source_target_table_used_as_child_weight": false`
(`src/neural_forge/lfm2_carrier_bridge_publication.py:927-928`, re-checked on
load at `:1058-1059`), and `expected_hidden_size` is used only as an assertion
on the child's own `(4096, H)` embedding shape (`:806`, checked at `:807-825`).
The A9/A11 rows in §1 are the
empirical proof: **the same two 1,933,888-element tensors already serve an
H1024 child (A9/A11) and an H816 child (r12/arm A).**

So the ~2x split is a subtraction, done once:

```
exact 2x counted total   229,693,184 / 2 = 114,846,592     (remainder 0 -- the parent count is even)
routes reserved                          -   3,867,776
                                          ------------
exact 2x dense body                       110,978,816
```

### 2.2 The exact total is attainable — no walk-down needed

Verified against the deployed solvers, CPU-only, read-only
(`/home/cereal/nf-w13t/probe_2x_lattice.py`, output `probe-2x.txt`;
`quarter_check.py`). The probe calls the production functions directly:
`enumerate_lfm2_global_hidden_sizes` (`src/neural_forge/hf_lfm2_cached_functional_width.py:617`)
then `derive_lfm2_automatic_mlp_width_policy`
(`src/neural_forge/hf_lfm2_cached_heterogeneous_capacity.py:2183`) — the same
two calls the global frontier makes at
`src/neural_forge/hf_lfm2_cached_global_topology_frontier.py:2227` and `:2248`
before it scores anything.

```
parent unique 229693184
exact quarter    total=57423296  body=53555520  ratio=4.0000000000  exact-lattice widths=76  816in=True  1024in=False
canonical 4x     total=57420144  body=53552368  ratio=4.0002195745  exact-lattice widths=57  816in=True  1024in=False
exact half       total=114846592 body=110978816 ratio=2.0000000000  exact-lattice widths=45  816in=True  1024in=True
```

Control first: at the sealed 4x body budget the probe reproduces the production
lattice — 125 legal widths, **57** with an exact lattice, **H816 present**
(and H840, which is what the frontier picks unpinned per
`tests/test_run_lfm2_dynamic_catalog.py:5193`).

At the exact half: **45 widths carry an exact lattice**, contiguous
`672, 680, …, 1016, 1024`. So:

* **the ~2x arm can be exactly 2.000000x** — unlike 4x, which sits 3,152 below
  the exact quarter, `--counted-target-total-parameters 114846592` needs no
  walk-down at all;
* **H816 is attainable at the 2x budget**, and so is H1024.

One honest correction to a standing assumption: the exact **quarter**
(`57,423,296` / body `53,555,520`) *also* has an exact MLP/operator lattice with
H816 (76 widths). So `57,420,144` was not forced by this lattice; it came from
somewhere upstream of it (option-bank/attention feasibility or the historical
frontier work — `docs/CURRENT_FOLDING_GOAL.md` records it as canonical without a
derivation). Not investigated further; flagged rather than smoothed over.

### 2.3 Hidden size — how 816 was actually derived, and the 2x equivalent

**816 was not derived by the topology solver.** It is a carrier ABI constant,
`LFM2_CARRIER_HIDDEN_SIZE = 816` (`experiments/run_lfm2_dynamic_catalog.py:347`,
mirrored as `CURRENT_LFM2_CARRIER_HIDDEN_SIZE` at
`src/neural_forge/lfm2_carrier_child_package.py:51`), and the config validator
*forces* `required_target_hidden_size` to equal it whenever carrier routes are
used (`:2207-2221`). The unpinned frontier at the 4x body budget picks H840
(`tests/test_run_lfm2_dynamic_catalog.py:5183-5196`). NOTES-1E blocker 1 already
records this: "the routed configuration is pinned to the real 4096-row H816
child".

So "derive the 2x equivalent the same principled way" resolves to: **keep 816.**

* It is attainable at the exact-2x body budget (§2.2), so nothing forces a move.
* Keeping it makes the 2x arm a **single-variable** move against arm A — budget
  only, same residual width, same basis machinery shape, same carrier ABI
  assertion — which is exactly what a ceiling map wants.
* Moving it is not free: `--carrier-expected-hidden-size 1024` is refused
  outright (§3, probe 5).

The alternative worth stating: **H1024** also has an exact lattice at 2x and
would make this arm the direct successor of the 1.35x same-body rung (body ratio
1.585x instead of 1.559x, zero residual contraction). It costs one extra guard
and moves two variables against arm A. **Recommendation: H816.** Do not pin
anything else — no `--balanced-basis-existing-phase-key-sha256`, no
`--global-topology-phase-key` (r11's fail-closed lesson stands).

---

## 3. The blocker — three carrier ABI constants, verified by refusal

The deployed runner refuses the composed arm at config time. Verified by
running the arm's **exact** flag set through the deployment's own
`config_from_args(build_parser().parse_args(argv))` — no model load, no GPU, no
filesystem write (`ls` afterward: `refusal-probe-run` does not exist).
Probe: `/home/cereal/nf-w13t/refusal_probe.py`, `refusal_chain.py`; transcripts
`refusal-probe.txt`, `refusal-chain.txt`.

```
=== 1. composed 2x carrier arm (2x budget, H816, carrier routes) ===
REFUSED ProductionLfm2CatalogError: Carrier-routed production requires the automatic
global heterogeneous factor-writer topology path.

=== 2. carrier routes + 4x budget constants, ratio 2 ===
REFUSED ProductionLfm2CatalogError: Carrier-routed production requires the automatic
global heterogeneous factor-writer topology path.

=== 3. same 2x arm with NO carrier route publication ===
REFUSED ProductionLfm2CatalogError: Carrier-routed folding requires the sealed route
publication, route artifact digest, semantic span evidence with both digests, and
routed manual prompt file with its digest.

=== 4. carrier + ratio 4 + 2x budget + H816 ===
REFUSED ProductionLfm2CatalogError: Carrier-routed production must reserve both
independent route channels inside the counted target and give only the
53,552,368-element remainder to the physical body.

=== 5. carrier + ratio 4 + 4x budget + H1024 ===
REFUSED ProductionLfm2CatalogError: The current authenticated carrier ABI requires an
untied 4096-row H816 child.

=== 6. CONTROL: arm A's exact numbers (ratio 4, 4x budget, H816) ===
ACCEPTED at config time.
  compression_ratio            4.0
  counted_target_total         57420144
  counted_dense_body           53552368
  dense_heterogeneous_target   53552368
  required_target_hidden_size  816
  uses_carrier_routes          True
```

Probe 6 is the control that matters: **the entire composed flag set is valid**;
only the three carrier ABI numbers refuse. Probe 2 isolates the ratio — with the
4x budget constants *unchanged*, `--compression-ratio 2` alone still refuses.

### 3.1 The guard sites (deployed tree == local tree for both files)

`experiments/run_lfm2_dynamic_catalog.py` sha256
`abad68f57c7bf62fd5306b8f97b14f7f1a1c1c675b0dbf24f6ba050fe4dc3cf1`,
`src/neural_forge/lfm2_carrier_child_package.py` sha256
`c7c537cf3f6ff0e3e35cdbf66acf9f81334912a81d81ad99b52c03900dcba28e` — identical
local and deployed, so the local line numbers are the deployment's.

| # | site | pins | fires for the 2x arm? |
|---|---|---|---|
| 1 | `run_lfm2_dynamic_catalog.py:2237-2252` | `math.isclose(compression_ratio, 4.0, rel_tol=0.0, abs_tol=0.0)` | **yes** |
| 2 | `run_lfm2_dynamic_catalog.py:2272-2287` | total `== 57_420_144`, body `== 53_552_368`, and `total - body == 2 x 1_933_888` | **yes** (first two clauses only) |
| 3 | `run_lfm2_dynamic_catalog.py:2288-2295` | `carrier_expected_hidden_size == 816`, rows `== 4096` | no, if the arm keeps H816 |
| 4 | `run_lfm2_dynamic_catalog.py:2207-2221` | `required_target_hidden_size == carrier_expected_hidden_size` | no |
| 5 | `run_lfm2_dynamic_catalog.py:11818-11845` + `lfm2_carrier_child_package.py:564-575` | runtime re-check via `current_exact4x_counted_plan()`, hard-wired to `CURRENT_LFM2_CARRIER_{TARGET,DENSE}_PARAMETER_ELEMENTS` | **yes**, later (post parent load) |
| 6 | `run_lfm2_dynamic_catalog.py:1638-1654` | carrier publication + span + `--routed-manual-prompts` are all-or-nothing | **yes**, if the carrier is dropped |

There is no escape flag. The deployed `--help` exposes exactly seven carrier
options and none of them relaxes a budget:

```
--carrier-expected-hidden-size  --carrier-expected-target-rows
--carrier-route-artifact-sha256 --carrier-route-publication
--carrier-span-evidence  --carrier-span-evidence-file-sha256  --carrier-span-evidence-sha256
```

### 3.2 Why "just drop the carrier routes" is not the arm

Guard 6 is an all-or-nothing coupling: dropping `--carrier-route-publication`
also forces dropping `--routed-manual-prompts`. Per NOTES-1E blocker 2, the
routed manual dossier, the routed smoke coverage, the counted-state verification
of the published child and the `manual_review_{started,completed}` stages all
sit behind `Lfm2CarrierChildPackageRegistration.current_exact4x(...).verify()`.

So a non-carrier 2x run **cannot produce a manual-review dossier at all** — and
"the intermediate-compaction arm scores nonzero on manual review" (R13_PLAN
item 4) is the entire deliverable. It would also move two variables at once
(compaction *and* token interface), destroying comparability with both the 1.35x
and 4x rungs, which were both measured through this same carrier package.

**This path is not a cheaper version of the arm. It is a different, uninformative
experiment.** Not taken.

### 3.3 The minimal change that unblocks it (not made)

This is the change NOTES-1E blocker 1 already specifies ("make the five ABI
numbers a named profile … plus one `toy` profile"), applied to a real second
profile instead of a toy one. Scoped, in the order the validator hits them:

1. `:2237-2252` — the exact-4.0 pin. In the heterogeneous path `compression_ratio`
   is **declarative only**: the budget request is built with
   `target_unique_parameters = dense_heterogeneous_target_parameters` and
   `match_policy="exact"` (`:10477-10486`). Replace the constant with a
   consistency check against `parent_unique / counted_target_total`, or drop the
   ratio clause and keep the three topology booleans.
2. `:2272-2287` — replace the two constant equalities with the invariant that is
   actually the ABI: `total - body == 2 * LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE`
   and `dense_heterogeneous_target_parameters == counted_dense_body`. Those two
   clauses are already in the same `if`; only the first two need to go.
3. `lfm2_carrier_child_package.py:564-575` — `current_exact4x_counted_plan()`
   parameterized on the requested total. The general
   `counted_plan(parent_parameter_elements=…, target_parameter_elements=…)` it
   wraps (`:552-563`) already does the right thing; only the *current* wrapper
   and its dense-body assertion are pinned. `_build_registered_carrier_counted_plan`
   (`run_lfm2_dynamic_catalog.py:953-978`) follows.
4. `:2288-2295` needs no change if the arm keeps H816.

Cost beyond the edit: four test assertions pin these messages and would need to
move with it —
`tests/test_run_lfm2_dynamic_catalog.py:1459` ("reserve both independent route
channels"), `:1496` ("equal carrier_expected_hidden_size"),
`tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py:1208` and `:1225`. Then the
usual chain: prove the tree at 9/9 on
`tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py` in an isolated clone,
bundle, cherry-pick into `neural-forge-postpub-e893767a` with a per-invocation
git identity, re-verify tree sha, and bump SOURCE_COMMIT — the NOTES-W13S §2
procedure exactly.

**Not made.** It is a source change to a fail-closed production guard on the
carrier lane, it needs review, and it is outside "compose and launch an arm".

---

## 4. The launcher that was composed but not written to the server

`nf-launch-r13-ceiling-2x.sh` was **not created on the server**, because a
launcher that cannot pass config validation is not a launcher.
`nf-launch-r12-round-refold.sh` was **not modified** — re-hashed this wave:

```
088dbe6b8521b0edc8a5d32319bcd7eb4b3a564c7948e7e0839e877b14f6a2c0  nf-launch-r12-round-refold.sh                          <- W13S post-edit form
6098d17904e95b6f677d4ae65681caee8277f92df4636fdca333618baf616873  nf-launch-r12-round-refold.sh.bak-w13s-20260808T082203Z <- W13S backup, untouched
```

The composition is fully determined and is recorded here so the next wave writes
it once. It is the refold stage of `nf-launch-r12-round-refold.sh` with the
staging machinery collapsed to a single `run` stage (keep the `SOURCE_COMMIT`
gate at lines 159-167, the dirty-checkout refusal, and the `nohup`/`setsid`
pattern), and these deltas:

**Removed**
```
--on-policy-round-manifest "$ROUND_MANIFEST"      <- base evidence only; the round's
--on-policy-max-rounds 2                             extension rollouts are the 4x child's,
                                                     off-policy for a 2x child, and dropping
                                                     them also avoids the merged-evidence lanes
--dense-closed-loop-relative-ridge-lambda 1e-5    <- the R-arm's two flags; the R-arm is the
--dense-closed-loop-body-max-prior-delta-ratio 8.0   arm that failed. Best-known == arm A.
```
plus the `round` and `preflight` subcommands, the `ROUND_PREIMAGE_PREFLIGHT`
block (no round manifest to read) and the `$R12` publication checks
(this arm reads no published child).

**Changed**
```
--counted-target-total-parameters      57420144  ->  114846592
--counted-dense-body-parameters        53552368  ->  110978816
--dense-heterogeneous-target-parameters 53552368 ->  110978816
--compression-ratio                           4  ->  2
--child-name  lfm2-integrated-carrier-exact4x-h816-mgv3-r12-round-0-refold
           -> lfm2-ceiling-2x-h816-mgv3-a0028
RUN=…/catalog-integrated-carrier/a0028-exact2x-h816-mgv3-ceiling      (NEW dir)
```

**Kept exactly** — base evidence, interface, and every arm-A solver flag:
```
--sealed-parent-cache-directory  $ROUND_BASE_CACHE   (base-cache-recaptured-w12e/c29b6737…)
--cached-parent-outputs          $REBOUND/canonical_parent_outputs_bridge_bound.json  (fa97d9a8…)
--tokenizer-bridge-mapped-parent-outputs $REBOUND/mapped_parent_outputs.json          (4fea48e1…)
  + the two REBOUND sha256 preflight checks and the parent_cache_key/sealed-rows probe
--carrier-route-publication / -artifact-sha256 / --carrier-span-evidence + both digests
--carrier-expected-target-rows 4096   --carrier-expected-hidden-size 816
--required-target-hidden-size 816     --dense-minimum-hidden-size 32
--dense-heterogeneous-layout / -global-hidden-frontier / -factor-writer-rank-discovery
--dense-heterogeneous-global-functional-rows-per-role 2048
--dense-residual-basis-mode circuit_balanced   --dense-run-closed-loop
--dense-body-reclosure-mode moving_gauge_v3
--dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement
--dense-moving-gauge-boundary-regression-tolerance 1e-6
--dense-moving-gauge-writer-norm-ratio-ceiling 2.0        <- defanged ceiling, unchanged
--dense-moving-gauge-install-verification fingerprint
--dense-closed-loop-replay-mode prefix_by_prefix_canonical
--dense-closed-loop-merge-adjacent-layer-captures
--dense-closed-loop-terminal-composition-floor
--dense-closed-loop-terminal-regression-tolerance 0.25
--dense-closed-loop-terminal-absolute-floor-minimum-layer 0
--dense-closed-loop-terminal-free-run-tripwire
--dense-closed-loop-input-error-multiplier 0              <- EIV 0, the control arm
--dense-closed-loop-solver-conditioning-diagnostics
--dense-closed-loop-gate-operating-point-diagnostics
--dense-closed-loop-depth-telemetry
--dense-streaming-vocabulary-margin-v2 (+ pair-chunk 256) --dense-vocabulary-margin-observability
--dense-attention-consumer-mode coupled-value-output      --dense-attention-geometry-risk
--dense-replay-devices cuda:0,cuda:4                      --dense-restart-capsule-keep 2
--max-selection-rows 13728 / --max-fit-rows 40656 / --max-validation-rows 13199
--routed-manual-* (1024 initial / 2048 retry, sealed prompt digest 0ee6bf14…)
phase caches ON, NO pinned keys
```

Phase caches: expect **cold** on both the global-topology and balanced-basis
entries. The topology phase key covers `exact_target_unique_parameters` and the
candidate width set (`hf_lfm2_cached_global_topology_frontier.py:2304-2330`),
both of which move at 2x; the balanced-basis entry is keyed to the evidence
identity and layout. That is correct — do not pin anything to force a hit.
Dropping the round manifest also returns the evidence set to **1 member** (the
recaptured base cache alone), not the 2-member set `f62110d7…` that arm A and the
R-arm carried.

---

## 5. Launch state, and the numbers that could not be measured

| | |
|---|---|
| launched | **no** |
| PID | none |
| run directory | none created (`a0028-exact2x-…` does not exist) |
| log | none |
| layer-0 terminal floor verdict | **not measured — the run never reached configuration acceptance** |

The L0 floor number the task wanted compared against arm A's `0.96988110` and
the R-arm's `1.00576305` does not exist for this arm and cannot be produced
without the §3.3 change. Stated plainly rather than substituted with an
estimate.

Server state, verified after all probing:

```
run_lfm2_dynamic_catalog processes                       0
all 7 GPUs                                               5-24 MiB, 0 % util
catalog-integrated-carrier/                              24 entries, unchanged
  arm A receipt.json      348,544 B  45bc68c2e8766e030516f6bccbe7ec2142c2af229e26454e44f3cfa71ecd4f10
  R-arm receipt.json                 5d61ae20913810f06324780e966d11522758352552d461312c549571ef40a1e7
deployment HEAD/TREE                 5739c742… / 48386cb0…, status empty
nf-launch-r12-round-refold.sh        088dbe6b…, 25,289 B, unmodified
```

Everything this wave wrote lives under `/home/cereal/nf-w13t/` and is read-only
tooling plus transcripts. No sealed directory, artifact, launcher, or peer run
was touched; nothing was deleted or renamed.

---

## 6. Monitoring recipe (for the arm once it is unblocked)

Unchanged from NOTES-W13S §7 except the run path and work dir. The W13S toolkit
under `/home/cereal/nf-w13s/` still works; repoint `RUN` and copy it forward.

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
W=/home/cereal/nf-w13t
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0028-exact2x-h816-mgv3-ceiling

$SSH --command-timeout 900   --execute "bash $W/poll.sh"                  # one-shot full poll
$SSH --execute "python3 $W/w13s_tl.py timeline"                           # stage timeline
$SSH --execute "python3 $W/w13s_tl.py basis"                              # balanced-basis hit/miss detail
$SSH --execute "python3 $W/r12r3.py $RUN"                                 # per-solve prior/ridge table
$SSH --execute "bash $W/ceiling.sh"                                       # ceiling evidence (post-body-reclosure only)
$SSH --command-timeout 15000 --execute "bash $W/waitlayers.sh 3 14000"    # block until N layers accepted
$SSH --execute "grep target_pid=<pid> $W/killspy-w13t.log"                # did anything signal it
```

Three things to expect that differ from the 4x arms:

* **The balanced basis will be built, not inherited.** Every 4x wave since W13D
  inherited `a0025`'s payload on phase key `52cbe3fd…`. At 2x that key rotates
  and the basis is cold. NOTES-W13S measured the 4x cold build at ~35 min; the
  2x layout is wider and 45 candidate widths survive the lattice instead of 57 —
  budget for a longer, not shorter, build, and do not treat a long silence there
  as a hang.
* **Layer 0's floor verdict is still the early signal.** The first
  `child_closed_loop_layer_started layer_index=0` → accept/refuse decision
  arrives ~11 min into the closed loop. Arm A's L0 terminal composed held-out
  relL2 was `0.96988110`; the R-arm's was `1.00576305` (refused). At a 1.559x
  body the L0 number should be materially below arm A's — **record it whatever
  it is; a floor refusal at layer 0 at ~2x would be a larger result than a
  clean pass.**
* **The floor/boundary/norm-ratio numbers still do not stream**, and on a
  *rejected* layer even the sealed artifacts carry nothing (NOTES-W13S §6.4,
  the open instrumentation defect). The first observable per-solve numbers
  arrive with the first accepted layer's restart capsule.

Re-arm killspy with its own 24 h window and its own log before launching, and
restart `rss-sampler.sh` / `milestones.sh` after any relaunch — both bind to
`run.pid` and self-exit with it.

---

## 7. Caveats

* **The arm did not run. There is no 2x data point in this file** — only the
  derivation that makes it launchable and the refusal that stopped it.
* **The §3 refusals are config-time only.** Guard 5 (the runtime
  `current_exact4x_counted_plan()` re-check, after the parent load) was reasoned
  from source, not triggered, because guards 1–2 fire first. If 1–3 are relaxed
  and 5 is missed, the run will fail closed minutes in rather than at config
  time — cheap, but it is the next thing to hit.
* **The exact-quarter finding in §2.2 contradicts an assumption, not a proof.**
  `57,423,296` having an exact MLP/operator lattice does not mean it was
  attainable end-to-end in 2026-07; the option-bank and attention feasibility
  layers were not probed. `57,420,144` stays canonical for 4x per
  `docs/CURRENT_FOLDING_GOAL.md`.
* **Which width the frontier would *select* unpinned at 2x is unknown.**
  Selection needs the parent cache and the functional-width evidence scan; the
  probe only established the feasible set. This is moot while the arm keeps
  H816, and it is the reason H816 is recommended rather than "let the solver
  decide".
* **The A9/A11 reconstructions in §1 are arithmetic, not sealed receipts.** They
  reproduce the two published ratios to six decimals, which is strong, but the
  sealed A9/A11 receipts were not re-read this wave.
* **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict.
* **Not committed.** No `git add`, no commit, no push. The only local file
  written is this one.

---

## 8. Artifacts

Local:

| path | |
|---|---|
| `NOTES-W13T.md` | this file |

Server, under `/home/cereal/nf-w13t/` (all read-only tooling, nothing else was written):

| file | |
|---|---|
| `probe_2x_lattice.py`, `probe-2x.txt` | §2.2's legal-width / exact-lattice derivation with the 4x control |
| `quarter_check.py` | exact quarter vs canonical 4x vs exact half, side by side |
| `refusal_probe.py`, `refusal-probe.txt` | §3's probes 1–3 — the composed arm's exact flag set through `config_from_args` |
| `refusal_chain.py`, `refusal-chain.txt` | §3's probes 4–6, including the ACCEPTED arm-A control |

Untouched and re-verified: `…/a0023-…-refold.sealed-armA-complete` (receipt
`45bc68c2…`, 348,544 B), `…-refold.failed-closed-terminal-floor-L0-w13s`
(receipt `5d61ae20…`), all other peer and partial run directories, the
deployment checkout, and `nf-launch-r12-round-refold.sh` (`088dbe6b…`).
