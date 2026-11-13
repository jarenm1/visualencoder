#!/usr/bin/env bash
# Run ON the Jetson Orin Nano (JetPack 6). Copy the deploy directories over first, e.g.
#   rsync -a deploy/baseline deploy/baseline-640 scripts/jetson_bench.sh jetson:visenc/
# then on the device:  cd visenc && bash jetson_bench.sh baseline baseline-640
# Builds FP16 and INT8 engines per directory and prints TensorRT's median GPU compute time.
# For stable numbers set max clocks first: sudo nvpmodel -m 0 && sudo jetson_clocks
set -euo pipefail
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

for dir in "$@"; do
  for precision in fp16 int8; do
    if [[ $precision == fp16 ]]; then onnx=encoder.onnx; flags=(--fp16); else onnx=encoder_int8.onnx; flags=(--int8 --fp16); fi
    engine="$dir/encoder_$precision.engine"
    log="$dir/trtexec_$precision.log"
    "$TRTEXEC" --onnx="$dir/$onnx" "${flags[@]}" --saveEngine="$engine" --useCudaGraph --iterations=200 --warmUp=500 >"$log" 2>&1
    median=$(grep -oP 'GPU Compute Time: .*median = \K[0-9.]+ ms' "$log" | tail -1)
    echo "$dir $precision: median GPU compute ${median:-see $log}"
  done
done
