#!/usr/bin/env bash
set -euo pipefail

source_root=/home/cereal/model-surgery-lab/neural-forge-src
artifact_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717
serial_root="$artifact_root/cached-causal35-rtx3090-gpu4-v1"
sharded_root="$artifact_root/cached-causal35-sharded-v1"
contract_document="$sharded_root/run-contract.json"
contract_pid_path="$sharded_root/logs/contract-gpu0.pid"
control_root="$sharded_root/control"
gpu6_release_file="$control_root/gpu6-probe.release"
venv_python=/home/cereal/model-surgery-lab/venv/bin/python

echo "orchestrator_waiting_for_contract=yes"
while [[ ! -f "$contract_document" ]]; do
  if [[ ! -f "$contract_pid_path" ]]; then
    echo "contract_failed=pid_file_missing"
    exit 1
  fi
  contract_pid="$(tr -dc '0-9' < "$contract_pid_path")"
  if [[ -z "$contract_pid" ]] || ! kill -0 "$contract_pid" 2>/dev/null; then
    echo "contract_failed=producer_exited_without_contract pid=${contract_pid:-missing}"
    exit 1
  fi
  sleep 5
done

cd "$source_root"
PYTHONPATH="$source_root/src:$source_root" \
  "$venv_python" - "$contract_document" <<'PY'
import json
import sys

from neural_forge.hf_lfm2_cached_causal_sharding import (
    verify_lfm2_cached_causal_run_contract,
)

path = sys.argv[1]
with open(path, encoding="ascii") as handle:
    contract = json.load(handle)
verify_lfm2_cached_causal_run_contract(contract)
assert contract["frontier"]["candidate_count"] == 84
assert contract["frontier"]["layer_count"] == 14
assert contract["frontier"]["target_fraction"] == 0.35
assert contract["forbidden_operations"] == {
    "parent_forward_calls": 0,
    "parent_cache_build_calls": 0,
    "parent_generation_calls": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
}
print(
    "contract_authenticated=yes "
    f"sha256={contract['run_contract_sha256']} "
    f"candidates={contract['frontier']['candidate_count']}"
)
PY

bash "$control_root/launch-probes.sh" 0 4

wait_for_probe() {
  local device_index="$1"
  local report_path="$sharded_root/probes/gpu$device_index.json"
  local pid_path="$sharded_root/logs/probe-gpu$device_index.pid"
  while [[ ! -f "$report_path" ]]; do
    if [[ ! -f "$pid_path" ]]; then
      echo "probe_failure=gpu$device_index-pid-missing"
      exit 1
    fi
    probe_pid="$(tr -dc '0-9' < "$pid_path")"
    if [[ -z "$probe_pid" ]] || ! kill -0 "$probe_pid" 2>/dev/null; then
      echo "probe_failure=gpu$device_index-exited-without-report"
      exit 1
    fi
    sleep 5
  done
}

wait_for_probe 0
wait_for_probe 4

echo "orchestrator_waiting_for_gpu6_release=$gpu6_release_file"
while [[ ! -f "$gpu6_release_file" ]]; do
  sleep 10
done
bash "$control_root/launch-probes.sh" 6
wait_for_probe 6

PYTHONPATH="$source_root/src:$source_root" \
  "$venv_python" - "$sharded_root/probes" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reports = []
for device in (0, 4, 6):
    with (root / f"gpu{device}.json").open(encoding="ascii") as handle:
        report = json.load(handle)
    body = dict(report)
    seal = body.pop("probe_sha256")
    payload = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert hashlib.sha256(payload).hexdigest() == seal
    assert report["clustering_calls"] == 0
    assert report["parent_forward_calls"] == 0
    assert report["parent_cache_build_calls"] == 0
    assert report["parent_generation_calls"] == 0
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0
    reports.append(report)

scientific_keys = (
    "run_contract_sha256",
    "option_id",
    "proposal_sha256",
    "canonical_pack_manifest_sha256",
    "canonical_pack_tensor_file_sha256",
    "canonical_bundle_sha256",
    "canonical_plan_sha256",
    "canonical_layer_binding_sha256",
    "canonical_medoids_sha256",
    "installed_child_identity",
    "installed_child_state_sha256",
)
reference = {key: reports[0][key] for key in scientific_keys}
for report in reports[1:]:
    assert {key: report[key] for key in scientific_keys} == reference
print(
    "cross_device_probe_authenticated=yes "
    f"option_id={reference['option_id']} "
    f"installed_child_identity={reference['installed_child_identity']}"
)
PY

serial_pid=""
if [[ -f "$serial_root/run.pid" ]]; then
  serial_pid="$(tr -dc '0-9' < "$serial_root/run.pid")"
fi
if [[ -n "$serial_pid" ]] && kill -0 "$serial_pid" 2>/dev/null; then
  record_root="$serial_root/work/candidate-records"
  baseline_count="$(find "$record_root" -maxdepth 1 -type f -name '*.json' | wc -l)"
  echo "serial_boundary_wait=yes pid=$serial_pid sealed_records=$baseline_count"
  while kill -0 "$serial_pid" 2>/dev/null; do
    current_count="$(find "$record_root" -maxdepth 1 -type f -name '*.json' | wc -l)"
    if (( current_count > baseline_count )); then
      echo "serial_boundary_observed=yes sealed_records=$current_count"
      kill -TERM "$serial_pid"
      break
    fi
    sleep 1
  done
  for _attempt in $(seq 1 30); do
    if ! kill -0 "$serial_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$serial_pid" 2>/dev/null; then
    echo "serial_stop_failed=yes pid=$serial_pid"
    exit 1
  fi
  final_count="$(find "$record_root" -maxdepth 1 -type f -name '*.json' | wc -l)"
  echo "serial_stopped_at_sealed_boundary=yes records=$final_count"
else
  echo "serial_already_stopped=yes pid=${serial_pid:-missing}"
fi

bash "$control_root/launch-shards.sh"

echo "orchestrator_waiting_for_manifests=yes"
while true; do
  manifest_count=0
  live_workers=0
  for shard_index in 00 01 02; do
    manifest="$sharded_root/shards/shard-$shard_index-of-03/shard-manifest.json"
    if [[ -f "$manifest" ]]; then
      manifest_count=$((manifest_count + 1))
    fi
  done
  if (( manifest_count == 3 )); then
    break
  fi
  for pid_path in "$sharded_root"/logs/shard-*-of-03-gpu*.pid; do
    [[ -e "$pid_path" ]] || continue
    worker_pid="$(tr -dc '0-9' < "$pid_path")"
    if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
      live_workers=$((live_workers + 1))
    fi
  done
  if (( live_workers == 0 )); then
    echo "shard_failure=no_live_workers manifests=$manifest_count"
    exit 1
  fi
  echo "shard_progress manifests=$manifest_count live_workers=$live_workers"
  sleep 30
done

bash "$control_root/launch-merge.sh"
echo "merge_dispatched=yes"
