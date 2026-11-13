#!/usr/bin/env bash
# Full baseline + ablation sweep. Run from the repo root inside the dev shell (direnv or `nix develop`).
# Teacher/student default to LingBot-Vision (Apache-2.0, ungated). DINOv3 runs are added
# automatically once your Hugging Face account has access to the gated facebook/dinov3-* repos.
set -euo pipefail

TEACHER=${TEACHER:-lingbot:large}
STUDENT=${STUDENT:-lingbot:small}
BATCH=${BATCH:-32}  # poses per step: ~1.6 GiB + 0.14 GiB/pose peak (32 -> ~6 GiB, 64 -> ~10.6 GiB)
STEPS=${STEPS:-20000}
# Spatial holdout: north-east quadrant never appears in training poses.
REGION=(0 0 245 245)

DS=facebook/dinov3-vits16-pretrain-lvd1689m
DB=facebook/dinov3-vitb16-pretrain-lvd1689m
DL=facebook/dinov3-vitl16-pretrain-lvd1689m
dinov3=false
if hf download "$DS" config.json >/dev/null 2>&1; then dinov3=true; else echo "DINOv3 not accessible yet; skipping DINOv3 runs"; fi

[[ -d assets/city ]] || { scene/fetch_assets.sh && blender -b --python scene/build_city.py -- --assets assets/kenney --out assets/city; }
[[ -d data/train ]] || visenc data --out data/train --seed 1 --count 20000 --random-views 2 --exclude-region "${REGION[@]}"
[[ -d data/eval ]] || visenc data --out data/eval --seed 2 --count 3000 --random-views 2

# A run is complete once its student checkpoint exists; a crashed run's directory is cleared and redone.
run() {
  local name=$1; shift
  [[ -d runs/$name/student ]] && return
  rm -rf "runs/$name"
  visenc distill --data data/train --steps "$STEPS" --batch-size "$BATCH" --out "runs/$name" "$@"
}
# Ablations (each changes one factor from `own-pretrained`).
run own-pretrained   --teacher "$TEACHER" --student "$STUDENT" --views random1 random2 --pair-target own
run mean-pretrained  --teacher "$TEACHER" --student "$STUDENT" --views random1 random2 --pair-target mean
run own-random       --teacher "$TEACHER" --student "$STUDENT" --views random1 random2 --pair-target own --student-init random
run own-noaug        --teacher "$TEACHER" --student "$STUDENT" --views random1 random2 --pair-target own --no-student-aug
run own-canonical    --teacher "$TEACHER" --student "$STUDENT" --views canonical       --pair-target own

encoders=(pixels lingbot-random:small lingbot:small lingbot:base lingbot:large
  run:runs/own-pretrained run:runs/mean-pretrained run:runs/own-random run:runs/own-noaug run:runs/own-canonical)
if $dinov3; then
  run dinov3-mean-pretrained --teacher "hf:$DB" --student "hf:$DS" --views random1 random2 --pair-target mean
  encoders+=("hf:$DS" "hf:$DB" "hf:$DL" run:runs/dinov3-mean-pretrained)
fi

# Jetson Orin Nano artefacts (ONNX for FP16, ModelOpt INT8 Q/DQ ONNX) and their simulated accuracy.
deploy_run() {
  local name=$1
  [[ -f deploy/$name/deploy.json ]] && return
  rm -rf "deploy/$name"
  visenc deploy --encoder "run:runs/$name" --calib data/train --out "deploy/$name"
}
for name in own-pretrained mean-pretrained; do
  deploy_run "$name"
  encoders+=("fp16:run:runs/$name" "int8:deploy/$name")
done

visenc eval --data data/eval --heldout-region "${REGION[@]}" --out reports/baselines --encoders "${encoders[@]}"
