# `run/mx22/queues/` — the wave-22 operational drivers, as they ran

These are the scripts that drove the wave from `/home/cereal/nf-mx22` on the
server. They are committed for the same reason the dossiers are: the wave's
numbers are only reproducible if the order the children were built and read in
is legible. `env.sh` names the one deployment every one of them sourced.

| queue | what it did |
|---|---|
| `parity.sh` | the conditioned capture against the sealed collector, before anything was built |
| `endpointq.sh` | the free measurement — the retained READ energy of every sealed basis **per endpoint**, which no census had printed, plus what the max-min iteration achieves at three widths. It runs FIRST and it can falsify step 3's hypothesis without a build |
| `basisq.sh` | one sealed basis artifact per (mode, width) the wave builds against |
| `widthq.sh` | the rungs, one build per rung against its own sealed basis |
| `reviewq.sh` | seal + frozen review, one child per invocation, refusing a tag it has already sealed |
| `../sealq.sh` | the ADDENDUM 6 verdicts, each a case-by-case reading of a frozen dossier |
| `../run-bench.sh` | the ADDENDUM 8 column, parent first |
