#!/usr/bin/env bash
# Exercise rendering plus external-memory handoffs with strict Vulkan validation.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
smoke=${1:-"$root/.sim-target/training/sim-smoke"}
: "${VK_LAYER_PATH:?Run inside the sim Nix development shell}"
test -r "$VK_LAYER_PATH/VkLayer_khronos_validation.json"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
status=0
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
    "$smoke" --frames 8 --output "$work/camera.ppm" > "$work/log" 2>&1 || status=$?
cat "$work/log"
if ((status != 0)); then
    exit "$status"
fi
# Some Vulkan drivers accept invalid SPIR-V after the layer reports an error.
if grep -Eq 'Validation Error|VUID-' "$work/log"; then
    echo 'FAIL: Vulkan validation reported an error' >&2
    exit 1
fi
echo 'PASS: Vulkan validation reported no errors'
