set -x
OC=/home/cereal/omni-campaign; U=$OC/composite/s7/units; S=$OC/composite/s7/scratch_dryrun
cd $OC/composite/tools
export CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src MEDIAFOLD_ROOT=$OC/media CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
PY=/home/cereal/model-surgery-lab/venv/bin/python
/usr/bin/time -v timeout 3000 $PY s7_dryrun.py --run-name A --build-child --scratch $S --units $U/ace/ace_dit_L12 $U/h3/h3_dit_L25_curve6 $U/hy3d/hy3d_dit_L10 --host128-ace $U/ace/ace_dit_L12
echo "RUN_A_EXIT $?"
/usr/bin/time -v timeout 3000 $PY s7_dryrun.py --run-name B --scratch $S --units $U/ace/ace_dit_L12 $U/h3/h3_dit_L25_curve6 $U/hy3d/hy3d_dit_L10 $U/qimg/qimg_mmdit_L30_img --host128-ace $U/ace/ace_dit_L12 --time-reference-rel-tol 5e-3
echo "RUN_B_EXIT $?"
