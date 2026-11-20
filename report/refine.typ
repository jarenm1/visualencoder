== Refining the encoder from flight failures <sec-refine>
The pilot is also a data engine: it flies realistic drone viewpoints (approaches, altitude changes, motion) that
random camera placement does not produce. We use its flights to find where the frozen encoder fails and refine the
encoder there, while guarding against overfitting to the simulator.

*An encoder gate.* A candidate is compared with a baseline encoder on three suites, each scored on held-out data with
paired 95% bootstrap intervals of the difference: (1) the simulator probes of @tab-main on shifted looks and the
held-out quadrant, plus retrieval and patch cosine (resampling poses); (2) fresh goal-matching heads trained on mined
flight frames from training districts and scored on a held-out district (resampling goals, since one goal's frames are
correlated); (3) UAVid sim→real and real→real readouts (resampling whole videos). The candidate is accepted only if the
failure score improves (interval above 0) and every other metric is non-inferior (interval above a fixed margin:
0.01–0.02). Selection never uses a score the training run optimizes.

*Encoder-centric failures.* A policy flies every episode of the held-out district; every 25 control steps the frame is
kept with the target box and the simulator's per-patch object labels (6,144 test frames). Failures are defined by a
fresh probe on frozen features, not by the policy. The paired student finds a visible target in 55% of frames and
correctly reports an absent one in 83%; the ViT-L teacher scores 53% and 82%. Of the student's 746 visible-target
errors, 466 land on another class, 162 on a same-class object of a different asset, 60 on background and 58 on
another instance of the same asset; false "in view" calls on absent targets are mostly same-class, different-asset
look-alikes (374 of 744). The teacher gets 575 of the student's failures right and also misses 915 of them.
Size dominates: a visible target covering 1–4 patches is found in 23% of frames, 5–12 in 43%, 13–30 in 57% and
larger targets in 69%. Upsampling the input 2× and pooling the finer tokens back does not help small targets
(1–4 patches: 19%), so finer tokens on the same pixels are not the missing ingredient.

*Refinement.* Starting from the paired student, the last four blocks are fine-tuned for 3,000 steps against the frozen
teacher through the student's original projection heads. Every step mixes UAVid training frames (own-view targets,
weighted 2×), simulator poses in two seen looks (the paired-view target: the teacher averaged over looks) and pairs of
flight frames. Two optional flight-frame losses target the failures directly: distillation weighted per patch by the
student–teacher gap, and a supervised contrastive loss over patch features in which patches of one layout object in
two frames of a flight are positives and every other object, same-class look-alikes included, is a negative.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: none, table.hline(),
    [*Candidate (vs. paired student)*], [*Failure score*], [*Car AP*], [*Car AUROC*], [*Real→real*], [*Retrieval*],
    table.hline(stroke: 0.5pt),
    [Task-head fine-tuning (@tab-goal)], [+0.017†], [−0.071], [−0.043], [+0.004], [−0.001],
    [Flight losses, no real anchor], [+0.007], [−0.070], [−0.036], [−0.004], [−0.035],
    [Flight losses + anchors], [−0.003], [+0.127], [+0.039], [+0.032], [−0.036],
    [Anchors only (own-view sim)], [+0.023], [+0.143], [+0.039], [+0.038], [−0.047],
    [*Anchors only (paired-view sim)*], [+0.016], [*+0.145*], [*+0.042*], [+0.035], [*−0.002*],
    table.hline(),
  ),
  caption: [Differences against the paired student (one refinement seed each). Failure score: fresh goal-matching
  head on held-out flight frames; real columns: UAVid sim→real car detection and real→real segmentation. Car AP for
  the paired-view anchors is +0.145 [0.095, 0.180]; its failure score interval is [−0.005, +0.038].
  † Measured on the earlier policy-mined frames, not the encoder-centric set.],
) <tab-refine>

@tab-refine separates the effects. The real anchor carries the real-image gains: without it, training on flight frames
lowers car AP and AUROC, the same pattern as fine-tuning through the task head. The flight-frame losses do not move
the failure score. Own-view simulator targets cost appearance invariance (retrieval −0.036 to −0.047); restoring the
paired-view target removes that cost. The refined student (paired-view anchors) raises sim→real car AP from 0.604 to
0.749 and AUROC from 0.903 to 0.944, real→real mIoU from 0.613 to 0.649 and sim→real mIoU from 0.516 to 0.537, while
every simulator readout stays within its margin. The gate still rejects it as a failure fix, because the failure score
gain is not established; we adopt it as the new base encoder on its guard results alone. The flight failures (small
objects and same-class look-alikes, which the teacher also misses) remain open.
