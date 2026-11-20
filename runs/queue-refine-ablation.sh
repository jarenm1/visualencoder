#!/usr/bin/env bash
# refine-v2 ablations, each gated against the current encoder (mean):
#   anchors-only   real/sim anchor distillation alone (no drone-frame losses)
#   no-real        v2 without the real-frame anchor
cd /home/jaren/visualencoder
P="nix develop --command nix develop path:sim --command env PYTHONPATH=sim/python:src .venv/bin/python -u"
LOG=reports/refine-ablation.log
run() {  # name, refine flags...
  echo "$(date +%T) refine $1" >> $LOG
  [ -f runs/refine-$1/summary.json ] || $P sim/python/train.py refine --mined runs/mined-drone13-train4/train \
    ${@:2} --out runs/refine-$1 >> $LOG 2>&1 || return
  $P sim/python/encoder_gate.py --baseline run:runs/mean --candidate run:runs/refine-$1 \
    --out reports/gate/refine-$1-vs-mean >> $LOG 2>&1
  status=$?
  echo "$(date +%T) gate $1 exit $status" >> $LOG
}
run anchors-only --hard-weight 0 --contrast-weight 0
run no-real --real-weight 0
