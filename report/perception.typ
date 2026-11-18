== Monocular depth and object heads
The drone's spatial memory places what the camera sees on a top-down map, which needs metric
depth from a single camera; counting needs per-patch object classes. Both are small heads on the
frozen 15×20 patch grid (1×1 projection to 192 channels, a 3×3 and a dilated 3×3 layer), supervised
by simulator labels that never reach the actor: per patch, the layout object most of its 16 samples
hit and the planar depth of that object's samples. Frames come from exploratory flight (8–60 m from
targets, randomized appearance and the monocular lens effects below), 16,384 for training and
3,072 from the held-out test district.

*Ground prior.* From the altimeter height and the gravity direction alone, each patch ray's distance
to the ground plane is known; three per-patch channels (log ground distance, ray elevation, whether
the ray reaches the ground) are concatenated to the tokens. Objects stand on the ground, so this
anchors metric scale without stereo.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: none, table.hline(),
    [*Encoder (frozen)*], [*Depth abs-rel* ↓], [*δ < 1.25* ↑], [*Class mIoU* ↑], [*Vehicle IoU* ↑],
    table.hline(stroke: 0.5pt),
    [LingBot ViT-S (off the shelf)], [0.204], [0.718], [0.714], [0.720],
    [Standard distillation], [*0.173*], [*0.786*], [*0.743*], [*0.763*],
    [Paired-view consistency], [0.178], [0.775], [0.732], [0.747],
    table.hline(),
  ),
  caption: [Held-out patch depth and class (none, building, vegetation, vehicle, prop). Without the ground prior
  the paired student's depth error is 0.214 (δ \< 1.25: 0.70). One seed per head; frames within a flight are correlated.],
) <tab-perception>

Distillation in the randomized city helps both readouts over the off-the-shelf model, matching the
simulator probes of @tab-main; the two distillation targets are within 0.01 of each other. With
labelled depth, the memory places a seen target within 3 m of its box in every case; with predicted
depth about half fall within 3 m, because a monocular range error grows with distance while the
bearing stays exact.

*Monocular camera effects.* The simulator's appearance randomization now also models a single
camera's lens and sensor timing per episode: radial distortion, vignetting, motion blur over the
exposure from the body rates, and rolling-shutter skew. Labels follow the same distortion.
