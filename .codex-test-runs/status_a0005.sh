BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/48e7913/a0005
for lane in source_producer_residual_writer_anchor_7 source_producer_residual_writer_anchor_0 full_strength_structural_anchor_0; do
  pid=$(cat "$BASE/logs/$lane.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state=running; else state=exited; fi
  artifacts=$(find "$BASE/$lane" -mindepth 1 -maxdepth 2 2>/dev/null | wc -l)
  logbytes=$(stat -c %s "$BASE/logs/$lane.log" 2>/dev/null || echo 0)
  echo "$lane pid=$pid state=$state artifacts=$artifacts logbytes=$logbytes"
  if [ "$state" = running ]; then ps -p "$pid" -o etime=,pcpu=,pmem=,rss=; fi
  if [ "$state" = exited ]; then tail -n 30 "$BASE/logs/$lane.log" 2>/dev/null; fi
done
for lane in source_producer_residual_writer_anchor_0 full_strength_structural_anchor_0; do
  pid=$(cat "$BASE/logs/$lane.resume.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state=running; else state=exited; fi
  mats=$(find "$BASE/$lane/materializations" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  dossier=$(stat -c %s "$BASE/$lane/whole-response-dossier.json" 2>/dev/null || echo 0)
  echo "$lane resume_pid=$pid resume_state=$state materializations=$mats dossier_bytes=$dossier"
  if [ "$state" = running ]; then ps -p "$pid" -o etime=,pcpu=,pmem=,rss=; fi
  if [ "$state" = exited ]; then tail -n 30 "$BASE/logs/$lane.resume.log" 2>/dev/null; fi
done
for lane in source_producer_residual_writer_anchor_7 source_producer_residual_writer_anchor_0 full_strength_structural_anchor_0; do
  pid=$(cat "$BASE/logs/$lane.resume2.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state=running; else state=exited; fi
  mats=$(find "$BASE/$lane/materializations-legacyfix" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  dossier=$(stat -c %s "$BASE/$lane/whole-response-dossier-legacyfix.json" 2>/dev/null || echo 0)
  echo "$lane resume2_pid=$pid resume2_state=$state materializations=$mats dossier_bytes=$dossier"
  if [ "$state" = running ]; then ps -p "$pid" -o etime=,pcpu=,pmem=,rss=; fi
  if [ "$state" = exited ]; then tail -n 30 "$BASE/logs/$lane.resume2.log" 2>/dev/null; fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
