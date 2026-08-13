#!/bin/bash
set -u
. /home/cereal/nf-mx21/env.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY $MX21/mx21_build.py --mode parity --device cuda:0
