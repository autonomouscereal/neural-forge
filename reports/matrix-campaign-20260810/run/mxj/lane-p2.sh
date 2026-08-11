#!/bin/bash
set -u
cd /home/cereal/nf-mxj
./drive.sh COMP.MA.MXJP2A M-A recipes/mxj-p2a.json 0
./drive.sh COMP.MA.MXJP2B M-A recipes/mxj-p2b.json 0
./drive.sh COMP.MA.MXJP2C M-A recipes/mxj-p2c.json 0
echo "P2 LANE COMPLETE"
