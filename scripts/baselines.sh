#!/usr/bin/env bash
# Full baseline + ablation sweep. Run inside `nix develop` after `huggingface-cli login`
# with access granted to the facebook/dinov3-* repos.
set -euo pipefail

S=facebook/dinov3-vits16-pretrain-lvd1689m
B=facebook/dinov3-vitb16-pretrain-lvd1689m
L=facebook/dinov3-vitl16-pretrain-lvd1689m
TEACHER=${TEACHER:-$B}
STEPS=${STEPS:-20000}
# Spatial holdout: north-east quadrant never appears in training poses.
REGION=(0 0 245 245)

[[ -d assets/city ]] || { scene/fetch_assets.sh && blender -b --python scene/build_city.py -- --assets assets/kenney --out assets/city; }
[[ -d data/train ]] || visenc data --out data/train --seed 1 --count 20000 --random-views 2 --exclude-region "${REGION[@]}"
[[ -d data/eval ]] || visenc data --out data/eval --seed 2 --count 3000 --random-views 2

run() { local name=$1; shift; [[ -d runs/$name ]] || visenc distill --data data/train --teacher "$TEACHER" --steps "$STEPS" --out "runs/$name" "$@"; }

# Ablations (each changes one factor from `own-pretrained`).
run own-pretrained   --views random1 random2 --pair-target own
run mean-pretrained  --views random1 random2 --pair-target mean
run own-random       --views random1 random2 --pair-target own --student-init random
run own-noaug        --views random1 random2 --pair-target own --no-student-aug
run own-canonical    --views canonical       --pair-target own

visenc eval --data data/eval --heldout-region "${REGION[@]}" --out reports/baselines --encoders \
  pixels "random:$S" "hf:$S" "hf:$B" "hf:$L" \
  run:runs/own-pretrained run:runs/mean-pretrained run:runs/own-random run:runs/own-noaug run:runs/own-canonical
