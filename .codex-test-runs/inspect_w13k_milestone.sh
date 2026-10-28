set -u
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0025-exact4x-h816-mgv3-r12-w13k-cold
PID=3262674
printf 'UTC_TIME '; date -u -Ins
ps -p "$PID" -o pid=,stat=,etime=,time=,%cpu=,%mem=,rss= || true
printf 'STAGE_COUNTS\n'
for pattern in \
  'checkpoint_plan' \
  'capsule' \
  'dense_structural_layer_completed' \
  'dense_structural_layer_started' \
  'layer_authenticated' \
  'layer_completed' \
  'fold_completed' \
  'error' \
  'failed'; do
  count=$(grep -ci "$pattern" "$RUN/run.log" 2>/dev/null || true)
  printf '%s %s\n' "$pattern" "$count"
done
printf 'CHECKPOINT_MATCHES\n'
grep -iE 'checkpoint[_ -]?plan|checkpoint.*tim|bulk.*checkpoint|capsule' "$RUN/run.log" 2>/dev/null | tail -30 || true
printf 'FIRST_AND_LAST_LAYER_COMPLETION\n'
grep '"stage":"dense_structural_layer_completed"' "$RUN/run.log" 2>/dev/null | sed -n '1p;$p' || true
printf 'LATEST_NON_PROGRESS_STAGES\n'
tail -80 "$RUN/run.log" 2>/dev/null | grep -v 'structural_tensor_' | tail -30 || true
