# The deployment every wave-19 queue sourced.  One deployment built and sealed
# every child in the wave; the two commits before it (`a69cfcc0`, `6cb1266a`)
# built nothing that survived -- the identity gate refused them both -- and were
# pruned under the current+1 policy.
export NF_DEPLOY=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-e7d068da
export W=/mnt/ai_data/neural-forge-matrix/wave19
export PY=/home/cereal/model-surgery-lab/venv/bin/python
export MX19=$NF_DEPLOY/reports/matrix-campaign-20260810/run/mx19
