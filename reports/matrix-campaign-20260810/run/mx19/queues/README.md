# `run/mx19/queues/` — the wave-19 operational drivers, as they ran

These are the scripts that drove the wave from `/home/cereal/nf-mx19` on the
server. They are committed for the same reason the dossiers are: the wave's
numbers are only reproducible if the order the children were built and read in
is legible. `env.sh` names the one deployment every one of them sourced.

| queue | what it did |
|---|---|
| `parity.sh` | the conditioned capture against the sealed collector, before anything was built |
| `censusq.sh` | the free residual-energy census and the sealed per-width basis artifacts |
| `widthq.sh` | the ladder rungs, one build per rung against its own sealed basis |
| `reviewq.sh` | seal + frozen review, one child per invocation |
| `../sealq.sh` | the seven ADDENDUM 6 verdicts, each a case-by-case reading of a frozen dossier |
| `../run-bench.sh` | the ADDENDUM 8 column, parent first |

**Three things this wave ran outside a queue and should be named.** The identity
gate (`--hidden-size 1024 --basis identity --arm stale --no-fold-solve
--no-head-gain --identity-gate report`) was run by hand three times, twice into
its own refusal; the 640 bisection rung was a basis write chained to a build
after 512 came back dead; and the quad was a plain wave-18-path excision build
(`--excise 1 --excise 5 --excise 9 --excise 13`) with no width axis at all.

**One operational mistake, recorded because the ledger guard caught it.** The
quad's review was queued on both GPUs, so it was sealed twice and its shard
carried two ARCH rows for one child. The merge kept the row the sealed verdict
names and dropped the other by digest; nothing was appended twice. The lesson is
the boring one: a review queue is not idempotent, and the guard that assumes it
is not is the one that earned its keep.
