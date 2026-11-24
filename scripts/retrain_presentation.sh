#!/usr/bin/env bash
# Launch from the repo root with: nix develop --command bash scripts/retrain_presentation.sh
# Use a new output directory per run. Single GPU: train/evaluate one stage at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/sim/python:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

seed=2
out=${1:-runs/presentation-retrain-seed$seed}
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/queue.log") 2>&1
base="run:$out/mean"
refined="run:$out/refined"
python=(.venv/bin/python -u)
sim=(nix develop path:sim --command "${python[@]}" sim/python/train.py)
heads_file() { local name=${1#run:runs/}; echo "$out/heads/${name//[\/:]/-}.pt"; }
stage() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

stage 'Distill: paired simulator views'
.venv/bin/visenc distill --data data/train-v2 --views random1 random2 --pair-target mean \
  --steps 10000 --seed "$seed" --out "$out/mean" > "$out/distill.log" 2>&1
stage 'Refine: simulator anchors and unlabeled UAVid training RGB'
"${python[@]}" sim/python/train.py refine --base "$out/mean" --mined runs/mined-drone13-train4/train \
  --steps 3000 --hard-weight 0 --contrast-weight 0 --seed "$seed" --out "$out/refined" > "$out/refine.log" 2>&1
stage 'Frozen-feature evaluation: simulation, real, and goal matching'
gate_status=0
"${python[@]}" sim/python/encoder_gate.py --baseline "$base" --candidate "$refined" \
  --out "$out/gate" > "$out/gate.log" 2>&1 || gate_status=$?
# Gate exit 1 means the candidate can miss the criteria; it is still a valid comparison.
if (( gate_status > 1 )) || [[ ! -s "$out/gate/verdict.json" ]]; then
  echo "Encoder evaluation failed; see $out/gate.log" >&2
  exit 1
fi

stage 'Perception heads: identical labeled frame split for both encoders'
"${sim[@]}" perception --encoders "$base" "$refined" --num-envs 32 \
  --train-frames 8192 --test-frames 3072 --epochs 16 --seed "$seed" \
  --out "$out/heads" > "$out/heads.log" 2>&1

for arm in mean refined; do
  enc="run:$out/$arm"
  run="$out/policy-$arm"
  mkdir "$run"
  stage "PPO ($arm): same warm start, matched 2M-step budget"
  "${sim[@]}" train --encoder "$enc" --goal patch --memory-heads "$(heads_file "$enc")" \
    --depth-view --explore-fraction 0.3 --mask-confusion --num-envs 32 --eval-envs 128 \
    --validation-catalog-size 8192 --encoder-precision fp16 --goal-refresh-steps 30 \
    --start-distance 8 60 --no-gps-fraction 0.5 --coarse-hint-fraction 0.3 --wrong-hint-fraction 0.1 \
    --init-from runs/drone13-v5-seed0/best.pt --total-steps 2000000 --validation-every 1000000 \
    --wall-seconds 0 --seed "$seed" --out "$run" > "$run/train.log" 2>&1
  stage "Held-out hard set ($arm): 256 episodes per cohort"
  "${sim[@]}" eval-hard --checkpoint "$run/best.pt" --verifier runs/verifier/test.pt \
    --eval-envs 256 --out "$run/hard-set.json" > "$run/hard-set.log" 2>&1
done
stage "Complete: $out/gate/verdict.md and $out/policy-{mean,refined}/hard-set.json"
