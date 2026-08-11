#!/bin/bash
# Wave 5B lane A (GPU 0): the short-conv input projection and the two attention
# read projections, each at the width --plan-storage-width said fits S4.
set -u
cd /home/cereal/nf-mxi
R=./run-mxi-cell.sh
$R C03.T1.S4.L00 L00-C03 0 192
$R C03.T2.S4.L00 L00-C03 0 192
$R C03.T3.S4.L00 L00-C03 0 192
$R C03.T5.S4.L00 L00-C03 0 728
$R C03.T8.S4.L00 L00-C03 0 192
$R C06.T1.S4.L12 L12-C06 0 128
$R C06.T2.S4.L12 L12-C06 0 128
$R C06.T3.S4.L12 L12-C06 0 128
$R C06.T5.S4.L12 L12-C06 0 242
$R C06.T8.S4.L12 L12-C06 0 128
$R C07.T1.S4.L12 L12-C07 0 85
$R C07.T2.S4.L12 L12-C07 0 85
$R C07.T3.S4.L12 L12-C07 0 85
$R C07.T5.S4.L12 L12-C07 0 121
$R C07.T8.S4.L12 L12-C07 0 85
echo "LANE A COMPLETE"
