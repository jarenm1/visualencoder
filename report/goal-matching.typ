== Goal matching on frozen encoders
In the full-city task the drone must find one object shown only as a goal crop, so the encoder's features must
separate look-alikes (two sedans, two trees), not only classes. A small head reads the frozen patch tokens and
predicts, per frame, a distribution over the 300 patches plus a "not in view" class; its expected position, peak
and pooled value feed the recurrent actor, and an auxiliary loss supervises it with the simulator's target box.

*Goal representations.* _Global_: the goal crop is one vector, its CLS token concatenated with its mean patch
token; each frame patch scores by a low-rank bilinear product with it. _Patch_: the goal crop's 15×20 patch
grid is pooled to 6×8, frame and goal patches share a learned rank-64 projection, and a frame patch scores by its
soft-max cosine similarity over the goal patches (learned temperature). "Not in view" also sees the best and mean
match, so absence is judged by "nothing here matches". The encoder stays frozen in both.

*Mined confusions.* A trained policy flies held-out episodes while every 0.5 s we label its frame: _wrong peak_
(target visible, policy confident, peak off the target) or _false view_ (target not visible, policy confident).
Held-out districts give 7,680 test frames; 10% are wrong-peak and 31–37% false-view failures of that policy. Each
frozen encoder gets a fresh head trained on 8,192 frames from training districts (identical seed and schedule).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: none, table.hline(),
    [*Encoder*], [*Global: visible* ↑], [*Global: score* ↑], [*Patch: visible* ↑], [*Patch: score* ↑],
    table.hline(stroke: 0.5pt),
    [LingBot ViT-S (off the shelf)], [0.493], [0.668], [0.621], [0.776],
    [LingBot ViT-L (teacher)], [0.538], [0.685], [*0.741*], [*0.819*],
    table.hline(stroke: 0.5pt),
    [Standard distillation], [0.610], [0.711], [0.722], [*0.819*],
    [Paired-view consistency], [*0.624*], [*0.730*], [0.706], [0.810],
    table.hline(),
  ),
  caption: [Held-out mined frames. Visible = peak patch on the target when it is visible; score = mean of that and
  correct "not in view" when it is not. One seed; frames within an episode are correlated, so no intervals are given.],
) <tab-goal>

@tab-goal shows that the goal representation matters more than the encoder. Patch matching raises every encoder
by 0.08–0.13 and improves both localization and absence. With a single goal vector the distilled students appear
to beat their teacher (0.730 vs 0.685); with patch matching that gap closes (teacher 0.819, students 0.810–0.819)
and the teacher localizes best. The students' advantage was in what survives averaging into one vector, not finer
discrimination. Fine-tuning the student's last four blocks on the mined frames through this head, anchored to the
teacher on real and simulated images, lowered real car AP (0.604 → 0.533). The encoder therefore stays frozen during
policy training; @sec-refine refines it separately, under a gate.

*Closed loop.* Two policies trained identically (frozen paired student, 2.0M transitions, final-3-second hold
success) differ only in the goal head. On 256 paired held-out episodes per cell, success with GPS rises from
2.7% (global) to 14.5% (patch) for standard 8–30 m starts, 0.4% to 2.0% when the target starts hidden, and 0.4%
to 1.6% for 35–50 m starts. Absolute success remains low: the 15 s horizon leaves no time to search, and the
policy has no memory beyond its recurrent state.
