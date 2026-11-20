#!/usr/bin/env bash
# (2) token-granularity test for small objects: the failure probe on mean vs up2:mean (no training);
# (1) anchors-only refinement with the original multi-view target on simulator frames, gated vs mean.
cd /home/jaren/visualencoder
P="nix develop --command nix develop path:sim --command env PYTHONPATH=sim/python:src .venv/bin/python -u"
LOG=reports/encoder-next.log
echo "$(date +%T) up2 probe" >> $LOG
$P sim/python/train.py confusion-probe --encoders run:runs/mean up2:run:runs/mean --mined runs/mined-drone13 \
  --held-out runs/mined-drone13 --heads patch global --out reports/resolution/probe.json \
  --samples reports/resolution/samples >> $LOG 2>&1
for head in patch global; do
  $P -m sim_rl.failures --split runs/mined-drone13/test --samples reports/resolution/samples \
    --encoders run:runs/mean up2:run:runs/mean --head $head --out reports/resolution/taxonomy-$head.json >> $LOG 2>&1
done
echo "$(date +%T) refine anchors-mv" >> $LOG
[ -f runs/refine-anchors-mv/summary.json ] || $P sim/python/train.py refine --mined runs/mined-drone13-train4/train \
  --hard-weight 0 --contrast-weight 0 --out runs/refine-anchors-mv >> $LOG 2>&1
$P sim/python/encoder_gate.py --baseline run:runs/mean --candidate run:runs/refine-anchors-mv \
  --out reports/gate/refine-anchors-mv-vs-mean >> $LOG 2>&1
status=$?
echo "$(date +%T) gate anchors-mv exit $status" >> $LOG
