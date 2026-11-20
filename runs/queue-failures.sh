#!/usr/bin/env bash
# Encoder-centric failure mining: drone13 flies (camera operator only), fresh probes on frozen
# student and teacher features label failures, the taxonomy classifies them. Waits for the gate.
cd /home/jaren/visualencoder
while pgrep -f queue-gate.sh > /dev/null; do sleep 60; done
P="nix develop --command nix develop path:sim --command env PYTHONPATH=sim/python:src .venv/bin/python -u"
OUT=runs/mined-drone13
LOG=reports/failures.log
echo "$(date +%T) mine" >> $LOG
$P sim/python/train.py mine --checkpoint runs/drone13-v5-seed0/best.pt --splits train test --num-envs 64 \
  --rounds 1 --horizons 2 --stride 25 --normal-rate 1 --out $OUT >> $LOG 2>&1 || exit 1
echo "$(date +%T) probes" >> $LOG
$P sim/python/train.py confusion-probe --encoders run:runs/mean lingbot:large --mined $OUT --held-out $OUT \
  --heads patch global --out reports/failures/probe.json --samples reports/failures/samples >> $LOG 2>&1 || exit 1
echo "$(date +%T) taxonomy" >> $LOG
for head in patch global; do
  $P -m sim_rl.failures --split $OUT/test --samples reports/failures/samples --encoders run:runs/mean lingbot:large \
    --head $head --out reports/failures/taxonomy-$head.json >> $LOG 2>&1 || exit 1
done
echo "$(date +%T) done" >> $LOG
