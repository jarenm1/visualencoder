#!/usr/bin/env bash
# Training-iteration throughput experiments (train_profile.py), one lever at a time.
cd /home/jaren/visualencoder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="nix develop --command nix develop path:sim --command env PYTHONPATH=sim/python:src .venv/bin/python -u sim/python/train_profile.py"
BASE="--encoder mean --goal patch --memory-heads runs/perception-v4/mean.pt --depth-view --explore-fraction 0.3 \
  --mask-confusion --goal-refresh-steps 30 --iterations 2"
LOG=reports/train-profile.log
run() {  # name, flags...
  echo "== $1" >> $LOG
  $P $BASE ${@:2} --out reports/train-profile-$1.json 2>&1 | grep -E "^n=" | cut -c1-400 >> $LOG
}
run e2-minibatch128 --envs 64 --minibatch-sequences 128
run e3-rollout32 --envs 64 128 --rollout-steps 32
run e1-256x192 --envs 64 128 --width 256 --height 192
run e1-192x144 --envs 64 128 256 --width 192 --height 144
run e1-128x96 --envs 64 128 256 512 --width 128 --height 96
echo "== done" >> $LOG
