#!/bin/bash
# Wave 5B lane B (GPU 4): the short-conv output projection, the attention value
# and output projections, and C10's two head-drop cells.
#
# C10.T1.S4.L02 is queued expecting a REFUSAL and is run anyway, because the
# head-drop kernel's own storage count -- which includes k_proj and v_proj in
# full, since a dropped query head does not stop the child storing them -- is
# the number ADDENDUM 6 6.2 binds, not the generic factor-pair model the
# planner falls through to for this component.  The refusal is the measurement.
set -u
cd /home/cereal/nf-mxi
R=./run-mxi-cell.sh
$R C05.T1.S4.L05 L05-C05 4 128
$R C05.T2.S4.L05 L05-C05 4 128
$R C05.T3.S4.L05 L05-C05 4 128
$R C05.T4.S4.L05 L05-C05 4 128
$R C05.T5.S4.L05 L05-C05 4 255
$R C05.T8.S4.L05 L05-C05 4 255
$R C08.T1.S4.L02 L02-C08 4 85
$R C08.T2.S4.L02 L02-C08 4 85
$R C08.T3.S4.L02 L02-C08 4 85
$R C08.T5.S4.L02 L02-C08 4 121
$R C08.T8.S4.L02 L02-C08 4 85
$R C09.T1.S4.L02 L02-C09 4 128
$R C09.T2.S4.L02 L02-C09 4 128
$R C09.T3.S4.L02 L02-C09 4 128
$R C09.T4.S4.L02 L02-C09 4 128
$R C09.T5.S4.L02 L02-C09 4 255
$R C09.T8.S4.L02 L02-C09 4 255
$R C10.T1.S4.L02 L02-C10 4 384
$R C10.T1.S2.L02 L02-C10 4 256
echo "LANE B COMPLETE"
