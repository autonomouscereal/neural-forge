#!/bin/bash
set -u
. /home/cereal/nf-mx23/env.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY $MX23/mx23_build.py --mode parity --device cuda:0
