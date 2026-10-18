# p2p11_card_guard.sh -- the card gate of lane p2p11's drivers (run_p2_v133.sh, run_p11_omni1_v110.sh).  SOURCED,
# never run.  1.0.0, 2026-10-14 (review finding: the drivers checked the card once, BEFORE a MemAvailable wait of up
# to 6 h, and read "no codex-d1-native* container" as "D1 is done" -- D1 went r4 -> r5 -> r6 on 2026-10-14, and a
# starting attempt held only ~360 MiB, under the 2,000 MiB card check).
#
# The caller defines GPU_INDEX, Q, stamp() and refuse() (refuse exits), and SLOT_FILE for GPU6.  It calls
#   gate_card <min-gib> <tag>   once, before the first GPU allocation: card_clear, the MemAvailable wait, card_clear
#   card_clear <when>           again right before EVERY later GPU job (each P2 cell, each P11 step)
#
# card_clear on GPU0 / GPU4 (the D1 cards) passes only when ALL of these hold at that moment:
#   - `docker ps` works and lists no codex-d1-native* container;
#   - the POSITIVE seal marker $Q/d1.sealed exists and is non-empty.  The lane never writes it: whoever finishes the
#     D1 seal (plan Step 4d, including bank v1, which itself runs on GPU0 + GPU4) writes it after Relay confirms
#     Codex launches no further attempt -- one line, the seal receipt path and the UTC time;
#   - nothing under runtime_leases/D1-* is newer than that marker (a D1 attempt started after the "seal");
#   - nothing under runtime_leases/D1-* changed in the last $P2P11_D1_QUIET_MIN minutes (default 30);
#   - the card holds < 2,000 MiB.
# On GPU6: $SLOT_FILE (written by the gpu6 lane owner) still exists, and the card holds < 2,000 MiB.
# Everything here is read-only on D1's container, lease dirs and rung dirs (docker ps, find, nvidia-smi).
#
# Test hooks (test_p2p11_card_guard.sh only; unset in production): P2P11_DOCKER, P2P11_NVSMI, P2P11_MEMINFO,
# P2P11_LEASES, P2P11_MEM_POLL_S, P2P11_MEM_WAIT_MAX_S, P2P11_D1_QUIET_MIN.
P2P11_GUARD_VERSION=1.0.0
P2P11_DOCKER=${P2P11_DOCKER:-docker}
P2P11_NVSMI=${P2P11_NVSMI:-nvidia-smi}
P2P11_MEMINFO=${P2P11_MEMINFO:-/proc/meminfo}
P2P11_LEASES=${P2P11_LEASES:-/home/cereal/omni-campaign/distill/runtime_leases}
P2P11_MEM_POLL_S=${P2P11_MEM_POLL_S:-120}
P2P11_MEM_WAIT_MAX_S=${P2P11_MEM_WAIT_MAX_S:-21600}
P2P11_D1_QUIET_MIN=${P2P11_D1_QUIET_MIN:-30}

mem_avail_gib() { awk '/MemAvailable/ {print int($2 / 1048576)}' "$P2P11_MEMINFO"; }

card_used_mib() {
  $P2P11_NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU_INDEX '$1==g{print $2}'
}

card_clear() {  # card_clear <when>: refuse unless GPU_INDEX is this lane's to use RIGHT NOW
  local when=$1 d1 fresh used marker=$Q/d1.sealed
  case $GPU_INDEX in
    0|4)
      d1=$($P2P11_DOCKER ps --filter name=codex-d1-native --format '{{.Names}}') \
        || refuse "[$when] docker ps failed: cannot prove D1 is off GPU0 / GPU4"
      d1=$(echo "$d1" | sed -n 1p)
      [ -z "$d1" ] || refuse "[$when] D1 container $d1 is running (it owns GPU0 + GPU4 until it exits)"
      [ -s "$marker" ] || refuse "[$when] no D1 seal marker $marker: no running container is not a sealed D1 (the D1 sealer writes it after the seal and Relay confirmation)"
      [ -d "$P2P11_LEASES" ] || refuse "[$when] lease root $P2P11_LEASES missing: cannot prove no D1 attempt is starting"
      fresh=$(find "$P2P11_LEASES" -mindepth 1 -maxdepth 2 -path "$P2P11_LEASES/D1-*" -newer "$marker" -print -quit) \
        || refuse "[$when] find over $P2P11_LEASES failed"
      [ -z "$fresh" ] || refuse "[$when] $fresh is newer than the D1 seal marker: a D1 attempt started after the seal"
      fresh=$(find "$P2P11_LEASES" -mindepth 1 -maxdepth 2 -path "$P2P11_LEASES/D1-*" -mmin -$P2P11_D1_QUIET_MIN -print -quit) \
        || refuse "[$when] find over $P2P11_LEASES failed"
      [ -z "$fresh" ] || refuse "[$when] D1 lease $fresh changed in the last $P2P11_D1_QUIET_MIN min: a D1 attempt may be starting"
      ;;
    6)
      [ -n "${SLOT_FILE:-}" ] && [ -f "$SLOT_FILE" ] || refuse "[$when] GPU6 slot file ${SLOT_FILE:-<unset>} missing (only the gpu6 lane owner grants it; never granted or withdrawn)"
      ;;
    *) refuse "[$when] GPU$GPU_INDEX is not a card of this lane" ;;
  esac
  used=$(card_used_mib) || refuse "[$when] nvidia-smi failed"
  [ "${used:-99999}" -lt 2000 ] 2>/dev/null || refuse "[$when] GPU$GPU_INDEX holds ${used:-?} MiB"
  echo "[$(stamp)] CARD CLEAR [$when] GPU$GPU_INDEX ${used} MiB$( [ "$GPU_INDEX" = 6 ] || echo "; D1 seal marker: $(sed -n 1p "$marker" | cut -c1-200)")"
}

gate_card() {  # gate_card <min-gib> <tag>: card_clear, wait for MemAvailable, card_clear AGAIN after the wait
  local need=$1 tag=$2 held=0
  card_clear "$tag:before-mem-wait"
  while [ "$(mem_avail_gib)" -lt "$need" ]; do
    [ $held -ge $P2P11_MEM_WAIT_MAX_S ] && refuse "MemAvailable stayed below $need GiB for $((P2P11_MEM_WAIT_MAX_S / 3600)) h"
    [ $((held % 1800)) -eq 0 ] && echo "[$(stamp)] MemAvailable $(mem_avail_gib) GiB < $need; waiting"
    sleep $P2P11_MEM_POLL_S; held=$((held + P2P11_MEM_POLL_S))
  done
  card_clear "$tag:after-mem-wait"
}
