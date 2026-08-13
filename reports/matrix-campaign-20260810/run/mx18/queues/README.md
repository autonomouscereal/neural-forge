# `run/mx18/queues/` — the wave-18 operational drivers, as they ran

These are the scripts that actually drove the wave from `/home/cereal/nf-mx18`
on the server: the parity check, the build queues in the order they were
submitted, the review queues, the chain that starts a review queue when a build
queue's marker line appears, and the benchmark queue.

They are committed for the same reason the dossiers are: the wave's numbers are
only reproducible if the order the children were built and read in is legible.
`env.sh` names the deployment every one of them sourced.
