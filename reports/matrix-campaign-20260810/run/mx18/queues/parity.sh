#!/bin/bash
set -u
. /home/cereal/nf-mx18/env.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY $MX18/mx18_build.py --mode parity --device cuda:0
