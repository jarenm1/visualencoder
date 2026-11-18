== End-to-end tasks
The full onboard stack (4-bit Gemma planner, frozen encoder, memory policy with the explore command) receives
one instruction per held-out free flight, without GPS and without a goal image: the instruction is all the
drone is told. Gemma replans every 5 simulated seconds from the camera frame and the memory summary;
a "go" decision names an object, which a separate grounding call boxes in the current frame, and the box
crop becomes the policy's goal. The policy flies every step. @tab-tasks reports 20 flights per task.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none, table.hline(),
    [*Grounding*], [*Count: exact / MAE*], [*Find: success (crashed)*], [*Explore: m² in 40 s (crashed)*],
    table.hline(stroke: 0.5pt),
    [Base Gemma], [5% / 4.2], [10% (55%)], [23,774 (20%)],
    [Grounding LoRA], [5% / 4.2], [0% (10%)], [26,385 (10%)],
    table.hline(),
  ),
  caption: [Count is scored against every car seen at any range over a 120 s flight (mean 6.9), while the
  onboard tracker counts within 35 m. Gemma's answer equals the memory's track count, so the count
  error is perception and memory recall, not language.],
) <tab-tasks>

Getting the loop to work at all took three fixes, none of them a flight rule:
- skills must keep the policy's recurrent state when the planner merely re-confirms them;
- the planner must never see simulator labels;
- at the deadline the only accepted action is an answer.

Find fails in two opposite ways. The base model boxes a look-alike (a building for "truck"), and the
drone often crashes into it. The first grounding adapter boxes almost nothing because of a data artifact:
repeated names were always phrased "the nearest X", so the agent's plain "the X" was learned as a refusal.
Crashes also end 10–20% of explore flights. Both point at the policy's held-out collision rate, which the
single-city runs do not fix.
