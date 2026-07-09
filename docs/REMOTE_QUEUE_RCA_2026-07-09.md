# Remote Queue RCA - 2026-07-09

## Issue

AI-server SSH sessions became unreliable while Neural Forge remote experiments
were being paused, inspected, and requeued.

The server was not CPU-saturated. On this host, process CPU values above 100%
represent multiple Linux scheduler threads, and the machine has many physical
cores. During the incident the host stayed responsive, total CPU remained below
full capacity, RAM was healthy after reboot, and swap was not active.

## Root Cause

The server was not CPU-bound. The control-plane failures came from SSH session
handling around queued background work.

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

The second failure mode was incomplete background detachment. Several launchers
redirected stdout/stderr to logs but did not explicitly close stdin. Under SSH,
long-lived background children can keep the remote channel tied to the invoking
session even when the host is otherwise healthy. Queue-launched children now
open stdin from `/dev/null`, write only to log files, and Python-spawned queue
jobs use `stdin=subprocess.DEVNULL`, `close_fds=True`, and a new session.

## Fix

- Reverted default CPU thread caps from experiment launch/eval scripts.
- Added `experiments/safe_process_control.py` for exact PID/tag controls.
- Added regression tests proving the helper does not select a self-matching
  shell command.
- Hardened Neural Forge launchers and queue fan-out jobs so background children
  close stdin with `/dev/null` and cannot pin the SSH channel.
- Added regression tests proving launchers and retest queues detach background
  stdin/session state.
- Added host capacity gates to `queue_beststack_after_retest_20260706.sh`:
  - `MIN_HOST_MEM_AVAILABLE_GB`
  - `MAX_HOST_SWAP_USED_GB`
  - `MAX_CONCURRENT_HEAVY_FOLDS`
- Kept per-GPU launch locks and CUDA capability guards.

## Operating Rules

- Do not use broad `pgrep -f ... | kill ...` one-liners for Neural Forge jobs.
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
