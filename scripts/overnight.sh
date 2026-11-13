#!/usr/bin/env bash
# Overnight queue: each run changes one factor from runs/baseline (LingBot-L -> LingBot-S, own target,
# seen views random1+random2, student aug on, 10k steps, seed 0). Ordered by priority; each run is
# evaluated right after training (reports/overnight/<name>), then one combined report at the end.
# Safe to re-run: finished runs and evals are skipped, crashed runs are redone.
set -uo pipefail
cd "$(dirname "$0")/.."

TEACHER=lingbot:large
DINOV3_L=hf:facebook/dinov3-vitl16-pretrain-lvd1689m
COMMON=(--data data/train-v2 --student lingbot:small --steps 10000 --batch-size 32)
REGION=(0 0 245 245)
mkdir -p reports/overnight

stamp() { date '+%F %T'; }

run() {
  local name=$1; shift
  if [[ ! -d runs/$name/student || ! -f runs/$name/log.jsonl ]] || ! grep -q '"step": 9999' "runs/$name/log.jsonl"; then
    rm -rf "runs/$name"
    echo "$(stamp) START $name: $*"
    if ! visenc distill "${COMMON[@]}" --out "runs/$name" "$@" > "reports/overnight/$name.train.log" 2>&1; then
      echo "$(stamp) FAILED $name (see reports/overnight/$name.train.log)"; return 0
    fi
    echo "$(stamp) TRAINED $name"
  fi
  if [[ ! -f reports/overnight/$name/report.md ]]; then
    visenc eval --data data/eval-v2 --heldout-region "${REGION[@]}" --out "reports/overnight/$name" \
      --encoders "run:runs/$name" > "reports/overnight/$name.eval.log" 2>&1 \
      && echo "$(stamp) EVAL $name: $(tail -1 "reports/overnight/$name/report.md")" \
      || echo "$(stamp) EVAL FAILED $name"
  fi
}

echo "$(stamp) queue start (GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader))"
run mean            --teacher "$TEACHER" --views random1 random2 --pair-target mean
run baseline-seed1  --teacher "$TEACHER" --views random1 random2 --pair-target own --seed 1
run canonical-only  --teacher "$TEACHER" --views canonical       --pair-target own
run dinov3-teacher  --teacher "$DINOV3_L" --views random1 random2 --pair-target own
run mean-seed1      --teacher "$TEACHER" --views random1 random2 --pair-target mean --seed 1
run random-init     --teacher "$TEACHER" --views random1 random2 --pair-target own --student-init random

encoders=(lingbot:small lingbot:large "$DINOV3_L" run:runs/baseline)
for name in baseline-seed1 mean mean-seed1 canonical-only dinov3-teacher random-init; do
  [[ -d runs/$name/student ]] && encoders+=("run:runs/$name")
done
echo "$(stamp) combined eval: ${encoders[*]}"
visenc eval --data data/eval-v2 --heldout-region "${REGION[@]}" --out reports/overnight/combined \
  --encoders "${encoders[@]}" > reports/overnight/combined.eval.log 2>&1 \
  && echo "$(stamp) DONE: reports/overnight/combined/report.md" || echo "$(stamp) combined eval FAILED"
