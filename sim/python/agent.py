"""Language agent: Gemma 4 reasons over the drone's memory and picks its next skill.

The learned policy flies; Gemma only decides what to do next. Every ``PLAN_SECONDS`` of simulated
time, or when the current skill ends, Gemma sees the current camera frame and a text summary of the
spatial memory (area looked at, confirmed car tracks with their offsets, the strongest goal
evidence) and answers with one JSON action:

  {"action": "explore"}                                    explore skill (policy explore command)
  {"action": "go", "box_2d": [ymin, xmin, ymax, xmax]}    fly to the object boxed in this frame
  {"action": "answer", "text": "..."}                     finish with an answer (e.g. a count)

Simulation time does not advance while Gemma thinks (the control loop waits), so planning latency
never costs flight time. Nothing here issues a flight command.
"""
from __future__ import annotations

import json
import re

from gemma_goal import ask, parse_box
from sim_rl.memory import CELL, EGO_RADIUS

PLAN_SECONDS = 5.0
TIME_LIMIT_SECONDS = 120.0
MAX_TOKENS = 128


def describe(offset):
    """'12 m ahead, 3 m left' for a (forward, right) offset in metres."""
    forward, right = offset
    along = f"{abs(forward):.0f} m {'ahead' if forward >= 0 else 'behind'}"
    across = f"{abs(right):.0f} m {'right' if right >= 0 else 'left'}"
    return f"{along}, {across}"


def summarize(memory, tracks, elapsed, skill):
    """Text the planner reads (environment 0): time, current skill, area looked at, cars and goal evidence."""
    seen_area = float(memory.map[0, 1].sum()) * CELL * CELL
    lines = [f"Time used: {elapsed:.0f} s of {TIME_LIMIT_SECONDS:.0f} s. Current skill: {skill}.",
             f"Ground area looked at so far: about {seen_area:.0f} square metres."]
    confirmed = tracks.confirmed(0)
    if confirmed:
        offsets = [describe(memory.relative(0, point)) for point in confirmed]
        lines.append(f"Cars seen so far: {len(confirmed)} ({'; '.join(offsets[:8])}).")
    else:
        lines.append("Cars seen so far: none.")
    candidate = memory.candidate()[0].tolist()
    if candidate[3] > 0.5:
        lines.append(f"Goal object last matched {describe((candidate[0] * EGO_RADIUS, candidate[1] * EGO_RADIUS))}.")
    return "\n".join(lines)


def parse_action(text):
    """First JSON object in ``text`` with a valid action, else None."""
    for match in re.finditer(r"\{.*?\}", text, flags=re.S):
        try:
            action = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        kind = action.get("action")
        if kind == "explore":
            return {"action": "explore"}
        if kind == "answer" and isinstance(action.get("text"), str):
            return {"action": "answer", "text": action["text"][:300]}
        if kind == "go":
            box = parse_box(json.dumps(action.get("box_2d")))
            if box is not None:
                return {"action": "go", "box": box, "label": str(action.get("label", ""))[:60]}
    return None


class Agent:
    def __init__(self, instruction, gemma, can_explore):
        self.instruction, (self.model, self.processor) = instruction, gemma
        self.can_explore = can_explore
        self.started_step = self.last_step = None
        self.skill, self.answer, self.history = "none", None, []

    def due(self, step, skill_ended):
        if self.answer is not None:
            return False
        return self.last_step is None or skill_ended or (step - self.last_step) >= PLAN_SECONDS * 30

    def prompt(self, summary):
        skills = ('"explore" (fly around and look), ' if self.can_explore else "") + \
            '"go" (fly to an object you box in this camera image), "answer" (finish and reply)'
        past = "; ".join(f"{h['action']}{' ' + h.get('label', '') if h.get('label') else ''}" for h in self.history[-6:])
        return (f"You are the planner of a camera drone. The user asked: \"{self.instruction}\".\n{summary}\n"
                f"Previous decisions: {past or 'none'}.\nSkills: {skills}.\n"
                "Choose the next step. Reply with one JSON object only, one of: "
                + ('{"action": "explore"}, ' if self.can_explore else "")
                + '{"action": "go", "box_2d": [ymin, xmin, ymax, xmax], "label": "..."} with coordinates 0-1000 '
                  'in this image, or {"action": "answer", "text": "..."}. Answer once you know the result or '
                  'when time is nearly up.')

    def plan(self, step, image, summary):
        """Ask Gemma for the next action. Returns (action or None, raw text, seconds)."""
        if self.started_step is None:
            self.started_step = step
        self.last_step = step
        text, latency = ask(self.model, self.processor, image, self.prompt(summary), MAX_TOKENS)
        action = parse_action(text)
        if action is not None and action["action"] == "explore" and not self.can_explore:
            action = None
        if action is not None:
            self.history.append(action)
            if action["action"] == "answer":
                self.answer = action["text"]
        return action, text, latency

    def elapsed(self, step):
        return 0.0 if self.started_step is None else (step - self.started_step) / 30
