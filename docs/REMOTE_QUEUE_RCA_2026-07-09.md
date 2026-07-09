# Remote Queue RCA - 2026-07-09

## Issue

AI-server SSH sessions became unreliable while Neural Forge remote experiments
were being paused, inspected, and requeued.

The server was not CPU-saturated. On this host, process CPU values above 100%
represent multiple Linux scheduler threads, and the machine has many physical
cores. During the incident the host stayed responsive, total CPU remained below
full capacity, RAM was healthy after reboot, and swap was not active.

## Root Cause

The server was not CPU-bound. `sshd` logs showed `MaxStartups` throttling: once
10 unauthenticated or stranded connections accumulated, later connections were
probabilistically dropped through connection 15. This continued for more than
an hour while host CPU and memory remained responsive. Parallel one-shot SSH
clients and commands that never returned amplified the pre-auth connection
pressure.

The first failure mode was broad command-line matching during manual process
control. Commands shaped like:

```bash
pgrep -f "$PATTERN" | xargs kill -STOP
```

can match the SSH-launched shell that contains the pattern text in its own
command line. If that shell is stopped, the SSH client waits on a remote command
that can no longer finish, which looks like an SSH/network stall even though the
server itself is still alive.

This same self-match shape appears in harmless status probes: a broad pattern
for `hf-xfamily-chain-anchor-compress` also matches the temporary shell running
the status command. That is safe when listing, but dangerous when combined with
`kill -STOP`, `kill -TERM`, or similar controls.

The second failure mode was incomplete background detachment. A live
reproduction showed that inherited stdout/stderr or PTY descriptors prevent the
SSH channel from reaching EOF. Redirecting stdin to `/dev/null` remains correct
hardening, especially with a PTY, but stdin alone was not the demonstrated
no-PTY blocker. Queue lock descriptor 9 could also survive into model processes
when a shell launcher inherited it.

## Fix

- Reverted default CPU thread caps from experiment launch/eval scripts.
- Added `experiments/safe_process_control.py` for exact PID/tag controls.
- Added regression tests proving the helper does not select a self-matching
  shell command.
- Added `experiments/launch_detached.py`, which starts jobs with
  `stdin=DEVNULL`, log-only output, `close_fds=True`, and
  `start_new_session=True`, then records PID, process start ticks, argv, and a
  command hash.
- Added `experiments/launch_qwen4b_path_run.py`, which requires an explicit
  target/run root, refuses reused run directories, snapshots the runner once,
  records target/config/script hashes, and launches through the detached helper.
- Updated the workstation Server Manager to serialize same-endpoint
  connections, retry pre-auth drops with backoff, drain stdout and stderr
  concurrently, close stdin/channels explicitly, and enforce a command timeout.
- Added regression tests proving launchers and retest queues detach background
  stdin/session state.
- Added host capacity gates to `queue_beststack_after_retest_20260706.sh`:
  - `MIN_HOST_MEM_AVAILABLE_GB`
  - `MAX_HOST_SWAP_USED_GB`
  - `MAX_CONCURRENT_HEAVY_FOLDS`
- Kept per-GPU launch locks and CUDA capability guards.

## Operating Rules

- Do not use broad `pgrep -f ... | kill ...` one-liners for Neural Forge jobs.
- Do not start parallel one-shot SSH clients to the same server. Batch related
  inspection commands through one Server Manager connection.
- Launch new long-lived experiments through `experiments/launch_detached.py` or
  a checked-in wrapper that uses it; do not rely on an outer SSH shell to call
  `setsid` correctly.
- Prefer PID files under `$RUN_ROOT/logs/*.pid`.
- If tag matching is needed, use:

```bash
/home/cereal/model-surgery-lab/venv/bin/python \
  experiments/safe_process_control.py list --tag "$TAG"
```

or for termination:

```bash
/home/cereal/model-surgery-lab/venv/bin/python \
  experiments/safe_process_control.py term --tag "$TAG" \
  --expect-contains hf-xfamily-chain-anchor-compress --wait-seconds 10
```

- Treat CPU thread caps as an explicit experiment-specific setting, not a
  default SSH reliability fix.
- Use the host capacity and heavy-fold concurrency gates to avoid memory/swap
  pressure when running multiple large donor folds.
