#!/bin/bash
set -u
. /home/cereal/nf-mx22/env.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY $MX22/mx22_build.py --mode parity --device cuda:0
