#!/bin/sh
# wait for the fit driver, then the all-real dry run + S5 plan check on the S5-bound units, then the summary
while ! grep -q "\[fit\] done" /home/cereal/omni-campaign/composite/s7/s5media/logs/drive_fit.log; do sleep 30; done
sh /home/cereal/omni-campaign/composite/s7/s5media/s5m_dryrun.sh /home/cereal/omni-campaign/composite/s7/units_s5 /home/cereal/omni-campaign/composite/s7/tap_rows_s5 /home/cereal/omni-campaign/composite/s7/s5media/scratch_dryrun_s5 S5ALL
CUDA_VISIBLE_DEVICES= nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python /home/cereal/omni-campaign/composite/s7/s5media/s5m_summary.py /home/cereal/omni-campaign/composite/s7 /home/cereal/omni-campaign/composite/s7/s5media/scratch_dryrun_s5 S5ALL /home/cereal/omni-campaign/composite/s7/s5media/S7_S5_MEDIA_UNITS.json
echo "final_done $(date -u +%FT%TZ)"
