OC=/home/cereal/omni-campaign; U=$OC/composite/s7/units; S=$OC/composite/s7/scratch_dryrun; LG=$OC/composite/s7/logs
cd $OC/composite/tools
export CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src MEDIAFOLD_ROOT=$OC/media CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
PY=/home/cereal/model-surgery-lab/venv/bin/python
/usr/bin/time -v timeout 2400 $PY s7_dryrun.py --run-name D --scratch $S --charted-dtype fp32 --units $U/ace/ace_dit_L12 $U/h3/h3_dit_L25_curve6 $U/hy3d/hy3d_dit_L10 $U/qimg/qimg_mmdit_L30_img $U/qvl32/qvl32_vit_L13 --host128-ace $U/ace/ace_dit_L12 --time-reference-rel-tol 5e-3
echo "RUN_D_EXIT $?"
timeout 600 $PY s7_dryrun_check.py --out $S/out_D --child $S/tiny_child > $LG/check_D.json 2>$LG/check_D.err
echo "CHECK_D_EXIT $?"
