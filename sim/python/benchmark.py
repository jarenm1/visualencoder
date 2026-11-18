"""Task benchmark: the whole onboard stack (Gemma agent + memory policy) on count, find and explore.

Each episode starts a fresh held-out free flight (live.py's Pilot, headless) without GPS and without
a goal image, gives the agent one instruction and runs until the agent answers, the drone crashes or
leaves the city, or the time limit passes. Gemma plans (simulated time waits while it thinks); the
policy flies every step.

  count    "How many cars are in this area? ..." -> the number in the answer ("no cars" = 0), against
           the distinct cars actually seen within the tracker's range during the flight (labelled
           object of >= 2 patches closer than MAX_RANGE_M in >= 3 frames, as count-eval; cars seen
           at any range are reported too). Also reports the memory's own track count at answer time.
  find     "Fly to the <name>." naming the episode's target -> success = the native reach-and-hold
           of that target before the time limit.
  explore  "Explore this area." -> ground area the memory marked looked-at after EXPLORE_SECONDS.

    nix develop path:sim --command .venv/bin/python sim/python/benchmark.py --gemma-4bit \\
        --checkpoint runs/drone11-v4-memory-explore-seed0/best.pt --out reports/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]

from agent import TIME_LIMIT_SECONDS  # noqa: E402
from gemma_goal import object_phrase  # noqa: E402
from live import Pilot  # noqa: E402
from sim_rl.counting import TRUE_PATCHES, VEHICLE  # noqa: E402
from sim_rl.memory import CELL  # noqa: E402
from sim_rl.perception import object_classes  # noqa: E402
from sim_rl.tracks import CONFIRM_HITS, MAX_RANGE_M  # noqa: E402

EXPLORE_SECONDS = 40.0
NUMBERS = {word: value for value, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
INSTRUCTIONS = {
    "count": "How many cars are in this area? Look around, then answer with the number of cars.",
    "explore": "Explore this area.",
}


def parse_count(text):
    """First number (digits or a word up to twelve) in ``text``; 0 for "no cars"-style answers; else None."""
    text = (text or "").lower()
    match = re.search(r"\b(\d+|" + "|".join(NUMBERS) + r")\b", text)
    if match is not None:
        word = match.group(1)
        return int(word) if word.isdigit() else NUMBERS[word]
    return 0 if re.search(r"\bno cars?\b|\bnone\b|(not|n't) (see|seen|found|find) any", text) else None


def run_episode(pilot, task, seed, lookup):
    pilot.new_episode(seed)
    pilot.last_outcome = None
    target_index = int(pilot.env.diagnostics[0, 27])
    target = pilot.object_info[target_index]["asset"]
    pilot.explore()   # no goal image or goal evidence: the instruction is all the drone is told
    instruction = f"Fly to the {object_phrase(target)}." if task == "find" else INSTRUCTIONS[task]
    pilot.instruct(instruction)
    # The agent's last plan, at TIME_LIMIT_SECONDS, only accepts an answer; explore stops earlier.
    limit = EXPLORE_SECONDS if task == "explore" else TIME_LIMIT_SECONDS + 1.0
    seen_frames = torch.zeros((2, lookup.shape[0]), device=pilot.device)   # within MAX_RANGE_M, any range
    agent, outcome, started = pilot.agent, "time limit", time.monotonic()
    while agent.elapsed(pilot.steps) < limit:
        before = pilot.last_outcome
        pilot.tick()
        objects = pilot.env.patch_object[0].long()
        vehicle = (objects >= 0) & (lookup[objects.clamp_min(0)] == VEHICLE)
        near = vehicle & (pilot.env.patch_depth[0] <= MAX_RANGE_M)
        hits = torch.zeros_like(seen_frames).scatter_add_(
            1, objects.clamp_min(0).expand(2, -1), torch.stack((near, vehicle)).float())
        seen_frames.add_((hits >= TRUE_PATCHES).float())
        if pilot.agent is None:   # crash or exit: tick reset the episode
            outcome = pilot.last_outcome if pilot.last_outcome != before else "ended"
            pilot.agent = agent
            break
        if task == "find" and pilot.last_outcome == "success" and before != "success":
            # Native success is for the current (possibly re-boxed) target; only the named one counts.
            outcome = "success" if int(pilot.env.diagnostics[0, 27]) == target_index else "wrong object"
            break
        if agent.answer is not None:
            outcome = "answered"
            break
    agent = pilot.agent
    row = {"task": task, "seed": seed, "instruction": instruction, "outcome": outcome,
           "seconds": round(agent.elapsed(pilot.steps), 1), "answer": agent.answer,
           "decisions": [h["action"] for h in agent.history], "wall_s": round(time.monotonic() - started, 1),
           "transcript": agent.transcript}
    if task == "count":
        seen = (seen_frames >= CONFIRM_HITS).sum(-1)
        row.update(truth=int(seen[0]), truth_any_range=int(seen[1]), estimate=parse_count(agent.answer),
                   memory_count=int(pilot.tracks.count()[0]))
    if task == "explore":
        row["seen_m2"] = round(float(pilot.spatial.map[0, 1].sum()) * CELL * CELL)
    pilot.agent = None
    return row


def summarize(rows):
    result = {}
    for task in sorted({row["task"] for row in rows}):
        part = [row for row in rows if row["task"] == task]
        crashed = sum(row["outcome"] in ("collision", "left the city", "ended") for row in part)
        summary = {"episodes": len(part), "crashed": crashed / len(part),
                   "answered": sum(row["answer"] is not None for row in part) / len(part)}
        if task == "count":
            scored = [row for row in part if row["estimate"] is not None]
            summary.update(
                parsed=len(scored) / len(part),
                mean_truth=sum(row["truth"] for row in part) / len(part),
                exact=sum(row["estimate"] == row["truth"] for row in scored) / len(part),
                within_one=sum(abs(row["estimate"] - row["truth"]) <= 1 for row in scored) / len(part),
                mean_abs_error=(sum(abs(row["estimate"] - row["truth"]) for row in scored) / len(scored)) if scored else None,
                memory_exact=sum(row["memory_count"] == row["truth"] for row in part) / len(part))
        if task == "find":
            summary["success"] = sum(row["outcome"] == "success" for row in part) / len(part)
        if task == "explore":
            summary["mean_seen_m2"] = sum(row["seen_m2"] for row in part) / len(part)
        result[task] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/drone11-v4-memory-explore-seed0/best.pt")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--gemma-4bit", action="store_true")
    parser.add_argument("--tasks", default="count,find,explore")
    parser.add_argument("--episodes", type=int, default=20, help="per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appearance", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="grounding LoRA (gemma_tune.py) used for the boxes")
    args = parser.parse_args()
    args.gemma, args.hz = True, 0.0
    pilot = Pilot(args)
    rows = []
    with pilot.env.execution():
        pilot.lanes[3] = 1.0   # no GPS: the policy gets no target offset, only the camera
        pilot.env.write_curriculum(pilot.lanes)
        lookup = object_classes(pilot.env)
        for task in args.tasks.split(","):
            for episode in range(args.episodes):
                row = run_episode(pilot, task, 1_000_003 * (args.seed + 1) + episode, lookup)
                rows.append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "transcript"}), flush=True)
    result = {"checkpoint": str(args.checkpoint), "gemma_4bit": args.gemma_4bit, "summary": summarize(rows), "rows": rows}
    print(json.dumps(result["summary"]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
