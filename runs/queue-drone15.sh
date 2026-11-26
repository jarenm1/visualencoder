#!/usr/bin/env bash
# Pilot on the new base encoder: perception heads for refine-anchors-mv, then drone15 = the drone13 recipe
# on that encoder (warm start from drone13, 48 envs), then its validation/test and hard set.
cd /home/jaren/visualencoder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="nix develop --command nix develop path:sim --command env PYTHONPATH=sim/python:src .venv/bin/python -u"
LOG=reports/drone15.log
ENC=run:runs/refine-anchors-mv
echo "$(date +%T) perception heads" >> $LOG
[ -f runs/perception-v5/refine-anchors-mv.pt ] || $S sim/python/train.py perception --encoders $ENC --train-frames 8192 \
  --test-frames 3072 --epochs 16 --out runs/perception-v5 >> $LOG 2>&1 || exit 1
R=runs/drone15-v5-refined-seed0; rm -rf $R; mkdir -p $R
echo "$(date +%T) drone15" >> $LOG
$S sim/python/train.py train --encoder $ENC --goal patch --memory-heads runs/perception-v5/refine-anchors-mv.pt \
  --depth-view --explore-fraction 0.3 --mask-confusion --num-envs 48 --eval-envs 256 --validation-catalog-size 8192 \
  --encoder-precision fp16 --goal-refresh-steps 30 --start-distance 8 60 --no-gps-fraction 0.5 \
  --coarse-hint-fraction 0.3 --wrong-hint-fraction 0.1 --init-from runs/drone13-v5-seed0/best.pt \
  --total-steps 8000000 --validation-every 1000000 --patience 3 --wall-seconds 0 --out $R > $R/train.log 2>&1
echo "$(date +%T) drone15 exit $?" >> $LOG
[ -f $R/summary.json ] || exit 1
$S sim/python/train.py eval-hard --checkpoint $R/best.pt --verifier runs/verifier/test.pt \
  --out $R/hard-set.json >> $LOG 2>&1
echo "$(date +%T) hard set exit $?" >> $LOG
