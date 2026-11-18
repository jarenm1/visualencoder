"""Onboard inference VRAM budget: the stack a drone would carry, without the simulator.

Loads each component in order on one GPU and records the device memory this process adds (NVML
per-process usage, so TensorRT engines count and other GPU users do not) plus the PyTorch peak
while it runs one control step or one Gemma answer. The goal is that the whole stack fits in 8 GB.

    nix develop path:sim --command .venv/bin/python sim/python/budget.py --quantize \
        --checkpoint runs/drone11-v4-memory-explore-seed0/best.pt --out reports/budget-4bit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pynvml
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]

from gemma_goal import detect, load_gemma  # noqa: E402
from sim_rl.cli import load_checkpoint  # noqa: E402
from sim_rl.features import checkpoint_features  # noqa: E402
from sim_rl.memory import SpatialMemory, observe as observe_memory  # noqa: E402
from sim_rl.model import GOAL_COMMAND, ActorCritic  # noqa: E402
from sim_rl.native import PRIVILEGED_WIDTH, PROPRIO_WIDTH  # noqa: E402

GIB = 1024 ** 3


def used():
    """Device memory held by this process (bytes)."""
    torch.cuda.synchronize()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    return sum(p.usedGpuMemory or 0 for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
               if p.pid == os.getpid())


class Ledger:
    def __init__(self):
        self.last = used()
        self.rows = [{"component": "CUDA context + PyTorch runtime", "device_gib": self.last / GIB}]

    def add(self, component, **extra):
        now = used()
        self.rows.append({"component": component, "device_gib": (now - self.last) / GIB,
                          "torch_peak_gib": torch.cuda.max_memory_allocated() / GIB, **extra})
        self.last = now
        torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--quantize", action="store_true", help="4-bit NF4 Gemma language model")
    parser.add_argument("--frame", type=Path, help="RGB image for the Gemma answer (default: grey 320x240)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pynvml.nvmlInit()
    device = torch.device("cuda:0")
    torch.zeros(1, device=device)
    ledger = Ledger()

    saved = load_checkpoint(args.checkpoint)
    meta = saved["metadata"]
    features = checkpoint_features(meta, device)
    frame = torch.rand(1, 3, meta["height"], meta["width"], device=device)
    cls, tokens = features.encode(frame)
    goal = features.goal_embedding(frame)
    ledger.add(f"encoder {meta['encoder']} ({meta['preprocessing'].split('/')[2]})")

    grid_h, grid_w = meta["height"] // 16, meta["width"] // 16
    policy = ActorCritic("visual", features.dim, features.goal_dim, grid_h, grid_w, meta["action_dim"], features.goal,
                         bool(meta.get("memory")), meta.get("commands", 1)).to(device).eval()
    policy.load_state_dict(saved["model"])
    proprio = torch.zeros(1, PROPRIO_WIDTH, device=device)
    proprio[:, 1] = -1.0   # level gravity
    starts, command = torch.ones(1, device=device), torch.full((1,), GOAL_COMMAND, device=device)
    memory = None
    if policy.spatial:
        spatial = SpatialMemory(1, grid_h, grid_w, meta["native_task_contract"]["camera"], device)
        memory = observe_memory(policy, spatial, cls, tokens, goal, proprio, starts)
    inputs, _ = policy.actor_inputs_and_features(cls, tokens, goal, proprio, torch.zeros(1, PRIVILEGED_WIDTH, device=device),
                                                 memory, command)
    policy.step(inputs, torch.zeros(1, policy.hidden_size, device=device), starts)
    ledger.add(f"policy ({sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M parameters, goal {features.goal}"
               + (", spatial memory and perception heads" if policy.spatial else "") + ")")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_gemma(args.model, args.quantize)
    ledger.add(f"{args.model} weights ({'4-bit NF4' if args.quantize else 'bf16'}; per-layer embedding table in host memory)")
    image = Image.open(args.frame).convert("RGB") if args.frame else Image.new("RGB", (320, 240), (128, 128, 128))
    _, _, latency = detect(model, processor, image, "Fly to the red car.")
    ledger.add("Gemma one grounding answer (activations, KV cache)", latency_s=round(latency, 3))

    total = sum(float(row["device_gib"]) for row in ledger.rows)
    report = {"checkpoint": str(args.checkpoint), "model": args.model, "gpu": torch.cuda.get_device_name(),
              "rows": ledger.rows, "total_device_gib": total, "fits_8_gib": total <= 8.0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in ledger.rows:
        print(f"{row['device_gib']:6.2f} GiB  {row['component']}")
    print(f"{total:6.2f} GiB  total (8 GiB budget: {'fits' if report['fits_8_gib'] else 'OVER'})")


if __name__ == "__main__":
    main()
