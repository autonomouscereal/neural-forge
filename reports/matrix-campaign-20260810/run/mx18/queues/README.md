# `run/mx18/queues/` — the wave-18 operational drivers, as they ran

These are the scripts that actually drove the wave from `/home/cereal/nf-mx18`
on the server: the parity check, the build queues in the order they were
submitted, the review queues, the chain that starts a review queue when a build
queue's marker line appears, and the benchmark queue.

They are committed for the same reason the dossiers are: the wave's numbers are
only reproducible if the order the children were built and read in is legible.
`env.sh` names the deployment every one of them sourced.

Wave 18 rewrote that `env.sh` line twice, because two of its findings changed the
builder mid-wave:

| queue | what it built | deployment |
|---|---|---|
| `tripleq.sh` | the four max-spread triples | `bf946174` |
| `lamq.sh` | `KV6JS`, `KV4JS` -- the lambda fix and the merge's second step | `bf946174` |
| `probeq.sh` | the gate probe, three children, control first | `bf946174`, then re-run on `c66c38ed` as `GATE2-*` once the logit regression was added |
| `mergeq.sh` | step 3's compositions, built BEFORE the mask defect was known | `bf946174`; all three were rebuilt |
| `causalq.sh` | the five bit-identity rebuilds and `X07KV6JQC` | `b372d47c` |
| `repairq.sh` | every composed child the defect could reach, plus the two width rungs | `b372d47c` |
| `reviewq.sh` | every dossier | whichever deployment sealed the child |

`probeq.sh` is committed as it last ran, with the `GATE2-` tags; the first run
used `GATE-` and the same flags, and both sets of per-site measurements are kept
in `wave18/probes/`.
