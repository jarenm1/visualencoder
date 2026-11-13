#!/usr/bin/env bash
# Frozen benchmark + one distillation baseline. Run from the repo root inside the dev shell.
# Later experiments (paired consistency, heads, real-image adaptation, QAT) fine-tune from
# runs/baseline and are evaluated on the same data/eval-v2 with the same command.
set -euo pipefail

TEACHER=${TEACHER:-lingbot:large}
STUDENT=${STUDENT:-lingbot:small}
BATCH=${BATCH:-32}    # poses per step: ~1.6 GiB + 0.14 GiB/pose peak (32 -> ~6 GiB)
STEPS=${STEPS:-10000} # loss plateaus around 10k steps (runs/own-pretrained log)
# Spatial holdout: north-east quadrant never appears in training poses.
REGION=(0 0 245 245)

[[ -d assets/city ]] || { scene/fetch_assets.sh && blender -b --python scene/build_city.py -- --assets assets/kenney --out assets/city; }
# v2 data: training sees only "seen" appearances; eval adds "novel" views (held-out textures and lighting).
[[ -f data/train-v2/meta.json ]] || { rm -rf data/train-v2; visenc data --out data/train-v2 --seed 11 --count 20000 --random-views 2 --exclude-region "${REGION[@]}"; }
[[ -f data/eval-v2/meta.json ]] || { rm -rf data/eval-v2; visenc data --out data/eval-v2 --seed 12 --count 3000 --random-views 2 --novel-views 2; }

if [[ ! -d runs/baseline/student ]]; then
  rm -rf runs/baseline
  visenc distill --data data/train-v2 --teacher "$TEACHER" --student "$STUDENT" --views random1 random2 \
    --pair-target own --steps "$STEPS" --batch-size "$BATCH" --out runs/baseline
fi
[[ -f deploy/baseline/deploy.json ]] || { rm -rf deploy/baseline; visenc deploy --encoder run:runs/baseline --calib data/train-v2 --out deploy/baseline; }

encoders=(pixels lingbot-random:small lingbot:small lingbot:large
  run:runs/baseline fp16:run:runs/baseline int8:deploy/baseline)
# Reference: the first run trained on v1 data, which included the now held-out textures.
[[ -d runs/own-pretrained/student ]] && encoders+=(run:runs/own-pretrained)
if hf download facebook/dinov3-vits16-pretrain-lvd1689m config.json >/dev/null 2>&1; then
  encoders+=(hf:facebook/dinov3-vits16-pretrain-lvd1689m hf:facebook/dinov3-vitl16-pretrain-lvd1689m)
fi

visenc eval --data data/eval-v2 --heldout-region "${REGION[@]}" --out reports/baseline --encoders "${encoders[@]}"
