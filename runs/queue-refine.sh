#!/usr/bin/env bash
# Post-train v2: more drone13 training flights, feature-level refinement, then the gate against
# the current encoder (mean) and the off-the-shelf student (lingbot:small).
cd /home/jaren/visualencoder
P="nix develop --command nix develop path:sim --command env PYTHONPATH=sim/python:src .venv/bin/python -u"
LOG=reports/refine.log
echo "$(date +%T) mine" >> $LOG
[ -d runs/mined-drone13-train4/train ] || $P sim/python/train.py mine --checkpoint runs/drone13-v5-seed0/best.pt \
  --splits train --num-envs 64 --rounds 4 --horizons 2 --stride 25 --normal-rate 1 \
  --out runs/mined-drone13-train4 >> $LOG 2>&1 || exit 1
echo "$(date +%T) refine" >> $LOG
[ -f runs/refine-v2/summary.json ] || $P sim/python/train.py refine --mined runs/mined-drone13-train4/train \
  --out runs/refine-v2 >> $LOG 2>&1 || exit 1
echo "$(date +%T) gate vs mean" >> $LOG
$P sim/python/encoder_gate.py --baseline run:runs/mean --candidate run:runs/refine-v2 \
  --out reports/gate/refine-v2-vs-mean >> $LOG 2>&1
echo "$(date +%T) gate vs mean exit $?" >> $LOG
$P sim/python/encoder_gate.py --baseline lingbot:small --candidate run:runs/refine-v2 \
  --out reports/gate/refine-v2-vs-small >> $LOG 2>&1
echo "$(date +%T) gate vs small exit $?" >> $LOG
