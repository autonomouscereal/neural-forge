#!/usr/bin/env bash
set -euo pipefail

artifact_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717
serial_root="$artifact_root/cached-causal35-rtx3090-gpu4-v1"
sharded_root="$artifact_root/cached-causal35-sharded-v1"

for label_pid in \
  "serial:$serial_root/run.pid" \
  "contract:$sharded_root/logs/contract-gpu0.pid" \
  "orchestrator:$sharded_root/logs/orchestrator.pid"
do
  label="${label_pid%%:*}"
  pid_path="${label_pid#*:}"
  if [[ -f "$pid_path" ]]; then
    pid="$(tr -dc '0-9' < "$pid_path")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      ps -p "$pid" -o pid=,etime=,stat=,%cpu=,%mem=,cmd= | sed "s/^/$label /"
    else
      echo "$label pid=${pid:-missing} live=no"
    fi
  else
    echo "$label pid_file=missing"
  fi
done

serial_records=0
if [[ -d "$serial_root/work/candidate-records" ]]; then
  serial_records="$(find "$serial_root/work/candidate-records" -maxdepth 1 -type f -name '*.json' | wc -l)"
fi
echo "serial_records=$serial_records"
serial_cache_dirs=0
if [[ -d "$serial_root/work/candidate-caches" ]]; then
  serial_cache_dirs="$(find "$serial_root/work/candidate-caches" -mindepth 1 -maxdepth 1 -type d | wc -l)"
fi
echo "serial_cache_dirs=$serial_cache_dirs"
if [[ -f "$serial_root/run.log" ]]; then
  grep -E '"event": "(candidate_measured|candidate_reused|candidate_cache_authenticated)"' "$serial_root/run.log" | tail -n 6 || true
fi
echo "contract_exists=$([[ -f "$sharded_root/run-contract.json" ]] && echo yes || echo no)"
canonical_pack_count=0
if [[ -d "$sharded_root/canonical-proposals" ]]; then
  canonical_pack_count="$(find "$sharded_root/canonical-proposals" -mindepth 2 -maxdepth 2 -type f -name manifest.json | wc -l)"
fi
echo "canonical_pack_count=$canonical_pack_count"
probe_report_count=0
if [[ -d "$sharded_root/probes" ]]; then
  probe_report_count="$(find "$sharded_root/probes" -maxdepth 1 -type f -name 'gpu*.json' | wc -l)"
fi
echo "probe_report_count=$probe_report_count"

for shard_index in 00 01 02; do
  shard_root="$sharded_root/shards/shard-$shard_index-of-03"
  shard_records=0
  if [[ -d "$shard_root/candidate-records" ]]; then
    shard_records="$(find "$shard_root/candidate-records" -maxdepth 1 -type f -name '*.json' | wc -l)"
  fi
  manifest_status=no
  if [[ -f "$shard_root/shard-manifest.json" ]]; then
    manifest_status=yes
  fi
  pid_status=missing
  for pid_path in "$sharded_root"/logs/shard-"$shard_index"-of-03-gpu*.pid; do
    [[ -e "$pid_path" ]] || continue
    shard_pid="$(tr -dc '0-9' < "$pid_path")"
    if [[ -n "$shard_pid" ]] && kill -0 "$shard_pid" 2>/dev/null; then
      pid_status="$shard_pid:live"
    else
      pid_status="${shard_pid:-missing}:exited"
    fi
  done
  echo "shard=$shard_index records=$shard_records manifest=$manifest_status pid=$pid_status"
done

if [[ -f "$sharded_root/logs/contract-gpu0.log" ]]; then
  tail -n 12 "$sharded_root/logs/contract-gpu0.log"
fi

gpu6_release_file="$sharded_root/control/gpu6-probe.release"
if [[ -f "$gpu6_release_file" ]]; then
  echo "gpu6_probe_release=yes path=$gpu6_release_file"
else
  echo "gpu6_probe_release=no path=$gpu6_release_file"
fi

nvidia-smi \
  --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
  --format=csv,noheader,nounits
nvidia-smi \
  --query-gpu=index,gpu_name,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
