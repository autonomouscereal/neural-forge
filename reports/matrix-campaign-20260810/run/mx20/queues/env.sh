# The deployment every wave-20 queue sources.  One deployment builds and seals
# every child in the wave; if a guard refusal forces a second commit, this file
# is the one place the new one is named, and the receipts carry it themselves.
export NF_DEPLOY=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-WAVE20HEAD
export W=/mnt/ai_data/neural-forge-matrix/wave20
export PY=/home/cereal/model-surgery-lab/venv/bin/python
export MX20=$NF_DEPLOY/reports/matrix-campaign-20260810/run/mx20
