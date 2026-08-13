#!/bin/bash
set -u
. /home/cereal/nf-mx20/env.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY $MX20/mx20_build.py --mode parity --device cuda:0
