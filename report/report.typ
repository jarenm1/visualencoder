// Draft project report (Type 1, 3-5 pages). Build: typst compile report/report.typ
#set document(title: "Appearance-Robust Vision Encoders for Drones via Sim Distillation", author: "Jaren McHugh")
#set page(paper: "us-letter", margin: 0.8in, numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show table: set text(size: 9pt)
#show figure.caption: set text(size: 9pt)

#align(center)[
  #text(size: 16pt, weight: "bold")[Appearance-Robust Vision Encoders for Drones \ via Distillation in a Randomized Simulator]
  #v(4pt)
  Jaren McHugh \
  #text(size: 9pt)[Code: `git.txstate.edu/<path>/visualencoder` (TODO)]
]

#block(inset: (x: 1.5em))[
  *Abstract.* Robots trained in simulation need visual features that stay stable when lighting and surface
  appearance change, the core of the sim-to-real gap. We distill a large self-supervised vision model
  (LingBot-Vision ViT-L, 300M parameters) into a small ViT-S (22M) small enough for a Jetson Orin Nano, using a
  procedurally built city in which only camera pose and appearance vary, so every image has exact depth,
  segmentation and pose labels. Standard distillation on randomized renders gives little robustness on its own.
  Adding a simple *paired-view consistency* target, where two differently lit and textured renders of the same pose
  share one teacher target, improves every metric under held-out textures and lighting (pose R² +0.11, depth error
  −17%, retrieval 0.75 → 0.95), and the 22M student then beats both 300M teachers on robustness. The model runs in
  INT8 with no measurable accuracy loss. On real drone footage (UAVid), simulator distillation improves sim-to-real
  segmentation transfer over the off-the-shelf small model and shrinks the sim-vs-real probe gap by about 40%, but the
  consistency gain does not carry over, which points to geometry and content, not only appearance, as the remaining gap.
]

= Overview
We build a vision encoder for a drone: a network that turns each camera frame into features that later modules
(a control policy, task heads, possibly a language model) consume. The encoder must be small enough for an
on-board Jetson Orin Nano, and its features must not change when only the appearance of the world changes
(sun, sky, fog, exposure, surface materials). That is a stand-in for the difference between simulation and
reality. We train entirely in simulation and measure robustness on appearance conditions never seen in training.

= Introduction
Large self-supervised vision models such as DINOv3 and LingBot-Vision produce strong general-purpose features but
are too large for small drones. Knowledge distillation trains a small student to reproduce a large teacher's
features. Domain randomization, which varies the appearance of simulated data, is the standard way to make
learned models robust to the sim-to-real gap.

Our question is whether combining the two yields a small encoder that is robust to appearance change, and what is
needed to get there. Our contributions:
+ A simulator and benchmark: one fixed city where only pose and appearance vary, with exact ground truth and a
  *held-out* appearance regime (unseen textures and lighting) plus a held-out region of the city.
+ A comparison of off-the-shelf encoders against distilled students under seen and unseen appearance.
+ A simple improvement, paired-view consistency, which turns randomization into robustness.
+ An INT8 deployment path to the Jetson Orin Nano that preserves accuracy.
+ A sim-to-real check on real drone footage (UAVid) that separates what transfers from what does not.

= Methodology

== Simulator and data
*Scene.* A Blender script assembles one 490 m city from free (CC0) Kenney asset kits: downtown towers, suburbs,
parks, roads, cars and a ring of forest. Geometry is fixed and seeded. It is exported as one model file per semantic
class (ground, road, building, vegetation, vehicle, prop; sky is the background).

*Renderer.* A custom raylib renderer draws three passes per camera pose: (1) lit RGB with Poly Haven textures
(applied in world space), normal maps and sun shadows, (2) per-pixel semantic class, (3) per-pixel metric depth.
Camera poses are sampled anywhere in the city at 2-45 m height with random yaw, pitch, roll and field of view,
avoiding collisions. Ground-truth state per image: position, orientation, field of view, distance to the nearest
object, and depth straight ahead.

*Appearance.* Only appearance is randomized: sun direction, colour and strength, sky and fog, exposure, colour tint,
and which texture each surface type uses. There are three regimes (@fig-looks): a fixed *canonical* look, a *seen*
distribution used for training, and a *novel* distribution used only for evaluation, with held-out textures (11 of
41), low sun, dim light, heavier fog and out-of-range exposure and tint.

*Splits.* Training: 20,000 poses, two seen looks each, excluding one quadrant of the city. Evaluation: 3,000
separate poses, each in canonical, two seen and two novel looks. Images are 320×240.

#figure(image("figures/looks.png", width: 85%), caption: [The same four poses in the canonical look, two seen
looks (training distribution) and two novel looks (held-out textures and lighting, evaluation only).]) <fig-looks>

== Distillation
The teacher is frozen LingBot-Vision ViT-L/16; the student is LingBot-Vision ViT-S/16 initialised from its
pretrained weights. Both output one global CLS vector and 300 patch vectors (a 15×20 grid of 16×16-pixel patches).
Each training step takes 32 poses × 2 seen looks. Both looks of a pose get the same random crop, so patches stay
aligned. The student sees an extra photometric corruption (colour jitter, blur, noise). Two linear heads project the
student's 384-d features to the teacher's 1024-d space. The loss is 1 − cosine similarity on the CLS token plus
the same on every patch.

*Baseline (standard distillation).* Each look matches the teacher's features of that same look.

*Improvement (paired-view consistency).* Both looks of a pose match the teacher's features *averaged over the two
looks*. The student must give one representation per physical place, whatever its appearance, while the target
remains the teacher's knowledge.

Training: 10,000 steps, AdamW (learning rate 2·10⁻⁴, warmup then cosine decay), bfloat16, about 70 minutes on one
RTX 5070 Ti.

== Evaluation
Features are frozen and read out with *linear* (ridge regression) probes, so scores reflect what the features
contain rather than probe capacity. Probes are fit on the canonical look of 80% of the evaluation poses and tested
on every look of the rest (442 poses). Metrics:
- *Pose R²*: global feature → state (position, pitch, roll, field of view, clearance). Yaw is excluded because it is
  only visible through sun direction, a lighting shortcut.
- *Depth*: per-patch log depth, absolute relative error (lower is better).
- *Segmentation*: per-patch class, mean IoU.
- *Retrieval\@1*: given a canonical image, find the same pose among all test poses in another look (no fitting).

*Real images.* UAVid (Lyu et al., 2020) contains real oblique drone frames of urban streets with per-pixel labels. We use its
labelled train (200 frames) and validation (70 frames, from different video sequences) splits, centre-cropped to 4:3
and resized to 320×240. Classes are mapped to ours (building, road, tree → vegetation, low vegetation → ground,
cars → vehicle); human and clutter have no counterpart in our city and are ignored. *Sim→real*: the segmentation probe
is fit on simulator renders only and applied unchanged to the real validation frames. *Real→real*: a probe fit on
UAVid train shows how much the features hold about real images at all; the difference between the two is the part of
the sim-to-real gap the features do not close.

== Deployment and tools
The student is exported to ONNX for a TensorRT FP16 engine. For INT8, NVIDIA ModelOpt inserts SmoothQuant
quantization on every linear layer, calibrated on 512 renders. LayerNorm inputs stay in FP16 because the residual
stream has outlier channels (values up to ±400) that the default configuration destroys (feature cosine 0.58).
Tools: PyTorch 2.14, HuggingFace Transformers, Blender, raylib, NVIDIA ModelOpt, ONNX Runtime, TensorRT; Nix and uv
for a reproducible environment.

= Results

== Baselines and improvement
@tab-main reports performance under the *novel* (held-out) appearance regime. Our rows average two training
seeds; the two seeds of each setting differ by at most 0.03 on any metric.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    [*Model*], [*Params*], [*Pose R²* ↑], [*Depth err.* ↓], [*Seg. mIoU* ↑], [*Retrieval* ↑],
    table.hline(stroke: 0.5pt),
    [LingBot ViT-S (off the shelf)], [22M], [0.345], [0.277], [0.675], [0.660],
    [LingBot ViT-L (teacher)], [300M], [0.375], [0.228], [0.788], [0.700],
    [DINOv3 ViT-L], [300M], [0.323], [0.228], [*0.835*], [0.775],
    table.hline(stroke: 0.5pt),
    [Standard distillation (baseline)], [22M], [0.465], [0.242], [0.788], [0.747],
    [*+ paired-view consistency (ours)*], [22M], [*0.574*], [*0.201*], [0.821], [*0.951*],
    table.hline(),
  ),
  caption: [Novel (held-out) textures and lighting. On the canonical look all distilled students score about the same
  (pose R² 0.63, segmentation 0.85), so the gains come from robustness rather than accuracy.],
) <tab-main>

Standard distillation already beats the off-the-shelf small model. Paired-view consistency improves every
metric under novel appearance (pose R² +0.11, depth error −17%, segmentation +0.03, retrieval +0.20) with no change on
the canonical look. With it, the 22M student beats both 300M models on pose, depth and retrieval, and is 0.014 behind
DINOv3-L on segmentation.

== Ablations
#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    [*Variant (one change from baseline)*], [*Pose R²*], [*Depth err.*], [*Seg. mIoU*], [*Retrieval*],
    table.hline(stroke: 0.5pt),
    [Baseline], [0.465], [0.242], [0.788], [0.747],
    [No appearance randomization], [0.506], [0.246], [0.769], [0.769],
    [DINOv3-L as teacher], [−0.134], [0.208], [0.838], [0.800],
    [Random-init student], [−0.239], [0.387], [0.534], [0.439],
    [Paired-view consistency], [*0.574*], [*0.201*], [0.821], [*0.951*],
    table.hline(),
  ),
  caption: [Novel appearance regime, single change from the baseline each.],
) <tab-ablation>

- *Randomization alone does not help:* training only on the canonical look is about as robust as the baseline.
  When each look copies the teacher on that look, the student also copies the teacher's sensitivity to appearance.
  Robustness appears only when looks of the same place are tied together.
- *The teacher's weaknesses are inherited:* a DINOv3 teacher gives the best segmentation, but pose under novel
  appearance collapses, as it does for DINOv3 itself.
- *Pretraining is essential:* a randomly initialised student is far worse everywhere.

== Real drone images
#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    [*Model*], [*Sim→real mIoU* ↑], [*Car AUROC* ↑], [*Real→real mIoU*], [*Gap* ↓],
    table.hline(stroke: 0.5pt),
    [LingBot ViT-S (off the shelf)], [0.492], [0.892], [0.650], [0.158],
    [LingBot ViT-L (teacher)], [0.561], [0.891], [0.712], [0.151],
    [DINOv3 ViT-L], [*0.574*], [0.849], [*0.736*], [0.162],
    table.hline(stroke: 0.5pt),
    [Standard distillation (ours)], [0.540], [*0.909*], [0.632], [0.093],
    [+ paired-view consistency (ours)], [0.526], [0.906], [0.615], [*0.090*],
    table.hline(),
  ),
  caption: [UAVid validation frames (real). Sim→real: segmentation probe fit on simulator renders only. Car AUROC:
  detecting patches with ≥10% car with the sim-fit probe. Real→real: probe fit on UAVid train. Gap = real→real −
  sim→real. Our rows average two seeds (seed spread ≈ 0.01–0.02).],
) <tab-real>

@tab-real and @fig-real test the encoders on real footage with a probe that never saw a real label.
- *Simulator distillation helps transfer.* The distilled ViT-S transfers better than the off-the-shelf ViT-S it
  started from (mIoU 0.492 → 0.540), detects cars best of all models (AUROC 0.91), and the gap between a sim-fit and a
  real-fit probe shrinks by about 40% (0.158 → 0.09): its features look more alike across sim and real.
- *Specialization has a cost.* Fit directly on real images, the students (0.62–0.63) fall slightly below the model they
  started from (0.650): training only on one stylised city trades away some general real-world quality. The large
  models remain best overall on real frames.
- *The consistency gain does not carry over.* Its large improvements under unseen simulated lighting and materials do
  not appear on real frames (0.526 vs 0.540, within seed noise). The real gap is evidently not only appearance:
  real scenes differ in geometry and content (fine detail, signage, people, varied architecture) from our low-poly city.

#figure(image("figures/real.png", width: 100%), caption: [Real UAVid frames: ground truth (black = classes with no
simulator counterpart) and predictions of segmentation probes fit only on simulator renders.]) <fig-real>

== Deployment
For the shipped model (paired-view consistency), INT8 matches FP32 (feature cosine 0.998) and every probe metric moves
by at most 0.009 (largest: novel-look retrieval 0.957 → 0.948); FP16 is identical. On-device latency on the Jetson is
still to be measured.

= Conclusion
Distilling a large vision model into a small one inside a randomized simulator does not by itself produce
appearance-robust features. Explicitly tying different looks of the same scene to one target does: it improves
robustness to unseen textures and lighting on every metric at no cost on the canonical look, and lets a 22M-parameter
encoder outperform 300M-parameter models under appearance shift while running in INT8 on an embedded GPU. On real
drone footage, simulator distillation improves transfer over the small model it starts from, but the consistency gain
stays in simulation: closing the rest of the sim-to-real gap needs more than appearance randomization.

*Limitations.* One stylised city and one renderer; linear probes measure what features contain, not downstream task
performance; two seeds per setting; real-image evaluation is segmentation only, on 70 frames of one dataset.

*Future work.*
+ Mix unlabelled real frames into distillation to keep general real-world quality while specializing.
+ Richer simulated geometry and content (detailed buildings, people, signage), not only appearance.
+ On-device latency of the INT8 engine on the Jetson Orin Nano.
+ An RL control policy on the frozen encoder, with temporal memory, in the drone simulator.
+ Task heads (depth, segmentation, obstacle clearance) and a foveated crop for distant targets.
+ A small language model (e.g. Gemma) reading the encoder's tokens for high-level goals.

#v(6pt)
#text(size: 9pt)[*References (to complete).* LingBot-Vision (Robbyant, 2026); DINOv3 (Meta, 2025); Hinton et al.,
Distilling the Knowledge in a Neural Network (2015); Tobin et al., Domain Randomization (2017); Xiao et al.,
SmoothQuant (2023); Lyu et al., UAVid: A Semantic Segmentation Dataset for UAV Imagery (2020, CC BY-NC-SA 4.0);
Kenney and Poly Haven asset libraries (CC0).]
