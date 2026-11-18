"""Language agent: Gemma 4 reasons over the drone's memory and picks its next skill.

The learned policy flies; Gemma only decides what to do next. Every ``PLAN_SECONDS`` of simulated
time, or when the current skill ends, Gemma sees the current camera frame and a text summary of the
spatial memory (time left, area looked at, confirmed car tracks with their offsets, the strongest
goal evidence) and answers with one JSON action:

  {"action": "explore"}                     explore skill (policy explore command)
  {"action": "go", "label": "red car"}      fly to that object; Gemma boxes it with the grounding
                                            prompt (gemma_goal.detect), "not visible" if it cannot
  {"action": "answer", "text": "..."}       finish with an answer (e.g. a count)

Planning and grounding are separate calls so a grounding adapter (gemma_tune.py) can serve the box
while planning uses the base weights. Once the time limit is reached the only accepted action is
answer. Simulation time does not advance while Gemma thinks (the control loop waits), so planning
latency never costs flight time. Nothing here issues a flight command.
"""
from __future__ import annotations

import contextlib
import json
import re

from gemma_goal import ask, detect
from sim_rl.memory import CELL, EGO_RADIUS

PLAN_SECONDS = 5.0
TIME_LIMIT_SECONDS = 120.0
MAX_TOKENS = 128
CONTROL_HZ = 30


def describe(offset):
    """'12 m ahead, 3 m left' for a (forward, right) offset in metres."""
    forward, right = offset
    along = f"{abs(forward):.0f} m {'ahead' if forward >= 0 else 'behind'}"
    across = f"{abs(right):.0f} m {'right' if right >= 0 else 'left'}"
    return f"{along}, {across}"


def summarize(memory, tracks, elapsed, skill):
    """Text the planner reads (environment 0): time, current skill, area looked at, cars and goal evidence."""
    seen_area = float(memory.map[0, 1].sum()) * CELL * CELL
    left = max(TIME_LIMIT_SECONDS - elapsed, 0.0)
    lines = [f"Time used: {elapsed:.0f} s; time left: {left:.0f} s. Current skill: {skill}.",
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


def parse_action(text, final=False):
    """First JSON object in ``text`` with a valid action, else None. ``final`` accepts only answers."""
    for match in re.finditer(r"\{.*?\}", text, flags=re.S):
        try:
            action = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        kind = action.get("action")
        if kind == "answer" and isinstance(action.get("text"), (str, int, float)):
            return {"action": "answer", "text": str(action["text"])[:300]}
        if final:
            continue
        if kind == "explore":
            return {"action": "explore"}
        if kind == "go" and isinstance(action.get("label"), str) and action["label"].strip():
            return {"action": "go", "label": action["label"].strip()[:60]}
    return None


class Agent:
    def __init__(self, instruction, gemma, can_explore):
        self.instruction, (self.model, self.processor) = instruction, gemma
        self.can_explore = can_explore
        self.started_step = self.last_step = None
        self.answer, self.history = None, []
        self.transcript = []   # (seconds, summary, raw reply) per plan

    @property
    def skill(self):
        """The current skill in the planner's own words (never simulator labels)."""
        last = next((h for h in reversed(self.history) if h["action"] in ("explore", "go")), None)
        if last is None:
            return "none"
        return "explore" if last["action"] == "explore" else f"go to the {last['label']}"

    def due(self, step, skill_ended):
        if self.answer is not None:
            return False
        if self.last_step is None or skill_ended or self.elapsed(step) >= TIME_LIMIT_SECONDS:
            return True
        return (step - self.last_step) >= PLAN_SECONDS * CONTROL_HZ

    def prompt(self, summary, final):
        past = "; ".join(h["action"] + (f" {h['label']}" if h.get("label") else "") + (" (not visible)" if h.get("missing") else "")
                         for h in self.history[-6:])
        head = (f"You are the planner of a camera drone. The user asked: \"{self.instruction}\".\n{summary}\n"
                f"Previous decisions: {past or 'none'}.\n")
        if final:
            return head + 'Time is up. Reply with one JSON object only: {"action": "answer", "text": "..."}.'
        skills = ('"explore" (fly around and look), ' if self.can_explore else "") + \
            '"go" (fly to an object you can see in this image), "answer" (finish and reply to the user)'
        return (head + f"Skills: {skills}.\nChoose the next step. Reply with one JSON object only, one of: "
                + ('{"action": "explore"}, ' if self.can_explore else "")
                + '{"action": "go", "label": "<object in this image>"}, or {"action": "answer", "text": "..."}. '
                  'Answer as soon as you know the result.')

    def planning(self):
        """Plan with the base weights when a grounding adapter is loaded."""
        disable = getattr(self.model, "disable_adapter", None)
        return disable() if disable is not None else contextlib.nullcontext()

    def plan(self, step, image, summary):
        """Ask Gemma for the next action -> (action or None, raw text, seconds). A ``go`` action
        carries the grounded ``box`` (None when Gemma cannot box the label in this frame)."""
        if self.started_step is None:
            self.started_step = step
        self.last_step = step
        final = self.elapsed(step) >= TIME_LIMIT_SECONDS
        with self.planning():
            text, latency = ask(self.model, self.processor, image, self.prompt(summary, final), MAX_TOKENS)
        self.transcript.append((round(self.elapsed(step), 1), summary, text))
        action = parse_action(text, final)
        if action is not None and action["action"] == "explore" and not self.can_explore:
            action = None
        if action is not None and action["action"] == "go":
            action["box"], grounding, seconds = detect(self.model, self.processor, image, f"Fly to the {action['label']}.")
            action["missing"] = action["box"] is None
            text, latency = f"{text}\n{grounding}", latency + seconds
            self.transcript[-1] = (*self.transcript[-1][:2], text)
        if action is not None:
            self.history.append(action)
            if action["action"] == "answer":
                self.answer = action["text"]
        elif final:
            self.answer = ""   # no parsable answer at the deadline: stop planning
        return action, text, latency

    def elapsed(self, step):
        return 0.0 if self.started_step is None else (step - self.started_step) / CONTROL_HZ
