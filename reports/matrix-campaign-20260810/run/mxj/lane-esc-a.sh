#!/bin/bash
set -u
cd /home/cereal/nf-mxj
./drive.sh COMP.MA.MXJ3A M-A recipes/mxj-s3a.json 0
./drive.sh COMP.MA.MXJ3B M-A recipes/mxj-s3b.json 0
./drive.sh COMP.MA.MXJ3C M-A recipes/mxj-s3c.json 0
./drive.sh COMP.MA.MXJ4C M-A recipes/mxj-s4c.json 0
echo "ESC-A LANE COMPLETE"
