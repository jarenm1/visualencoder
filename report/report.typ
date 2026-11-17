// Project report (Type 1, 3-5 pages). Build: typst compile report/report.typ report.pdf
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
  Jaren McHugh
]

#block(inset: (x: 1.5em))[
  *Abstract.* We distill a 300M-parameter vision model into a 22M student using randomized city images.
  Sharing teacher targets between appearance variants improves novel-appearance readouts, but does not
  by itself improve real UAVid precision. Matched two-seed interventions separate layout variation from
  unlabelled-real exposure: additional layouts within a shared asset library add little at fixed compute,
  while mixing real RGB improves segmentation and car average precision with little loss of simulated
  robustness. AUROC and probe-gap rankings disagree with precision and absolute quality. Whole-video
  bootstrap intervals expose uncertainty, with only seven validation videos. A separate Vulkan/NVIDIA Rust CUDA
  pipeline trains frozen-encoder recurrent policies within an hour, but an actual-city pilot remains below
  80% success; neither convergence nor an encoder-ranking advantage is established. INT8 preserves measured probe scores, but embedded-device latency is unmeasured.
]

= Introduction
Large self-supervised vision models such as DINOv3 and LingBot-Vision produce strong general-purpose features.
Knowledge distillation trains a small student to reproduce a large teacher's features, reducing model size.
Domain randomization varies simulated appearance to improve robustness to distribution shift.

Our question is whether combining the two yields a small encoder that is robust to appearance change, and what is
needed to get there. Our contributions:
+ An appearance benchmark and an independently matched layout × consistency experiment.
+ Paired-target distillation compared against pretrained encoders and standard distillation.
+ Unlabelled-real adaptation evaluated with car AP and whole-video uncertainty, not AUROC alone.
+ A GPU-owned control pipeline with timed PPO, blind/privileged controls and a separate INT8 probe check.

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
- *Pose R²*: global feature → state (position, pitch, roll, field of view, clearance). Yaw is excluded from
  this aggregate to avoid letting sun-direction shortcuts dominate its interpretation.
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

*Precision.* An ONNX/SmoothQuant export of the original simulation-only paired student changes probe scores
by at most 0.009. Calibration and precision checks are in the supplement; Jetson latency is unmeasured.

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

== Real drone images
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    [*Model*], [*Sim→real mIoU* ↑], [*Car AUROC* ↑], [*Car AP* ↑], [*Real→real mIoU*], [*Gap*],
    table.hline(stroke: 0.5pt),
    [LingBot ViT-S (off the shelf)], [0.492], [0.892], [0.640], [0.650], [0.158],
    [LingBot ViT-L (teacher)], [0.561], [0.891], [*0.727*], [0.712], [0.151],
    [DINOv3 ViT-L], [*0.574*], [0.849], [—], [*0.736*], [0.162],
    table.hline(stroke: 0.5pt),
    [Standard distillation (ours)], [0.539], [*0.909*], [0.649], [0.632], [0.093],
    [Paired-view consistency (ours)], [0.525], [0.905], [0.622], [0.615], [0.090],
    table.hline(),
  ),
  caption: [UAVid validation. Sim→real uses a simulator-fit probe; real→real uses a UAVid-train probe.
  Car AUROC/AP detect patches with ≥10% car versus car-free patches. Gap = real→real − sim→real.
  Student rows average two seeds. DINOv3 point estimates are original; new AP/CI results are unavailable (gated weights).],
) <tab-real>

@tab-real tests transfer with a simulator-fit probe; supplementary figures show qualitative predictions.
- *Simulator distillation helps transfer.* The distilled ViT-S improves sim→real segmentation over the
  off-the-shelf ViT-S (mIoU about 0.49 → 0.54) and car AUROC (0.892 → 0.909). The sim-fit versus real-fit probe
  gap shrinks, but this is not sufficient evidence by itself: lower real→real quality also reduces that gap.
- *Specialization has a cost.* Fit directly on real images, the students (0.62–0.63) fall slightly below the model they
  started from (0.650): training only on one stylised city trades away some general real-world quality. The large
  models remain best overall on real frames.
- *Simulation-only consistency does not add a real-image gain.* Its large gains under unseen simulated lighting and materials do
  not appear on real frames (about 0.53 vs 0.54, within seed variation). Geometry and content mismatch are plausible
  explanations; the original fixed-city experiment does not isolate their contribution.

*Car precision and uncertainty.* Across the two training seeds, car average precision is 0.649 for standard
distillation and 0.622 for paired consistency, versus 0.640 off the shelf. There are 1,671 positive and 17,801
car-free eligible patches (8.58% positives); ambiguous patches are excluded. AUROC gives tied scores half credit,
and AP groups tied thresholds. With 1,000 whole-video bootstrap resamples, the standard student's AP improvement
over the small baseline has a paired 95% interval of −0.038 to +0.044 for seed 0 and −0.046 to +0.058 for seed 1.
Thus the higher AUROC is not a clear improvement in AP. Only seven validation sequences support these intervals;
they exclude model-training and probe-fitting uncertainty.
AP also reverses the AUROC ranking between the large LingBot teacher and the students.


#include "experiments.typ"

*Cross-intervention comparison.* Extra training layouts give only small dense-readout changes and no
consistent real-transfer gain at this budget. They retain the same asset library, so this is not a test of
arbitrary real-world geometry or content. Global readouts and real-transfer effects are in the supplement.

In the mixed condition, real-image readout quality and car AP improve, without an AUROC improvement.
For paired targets, the matched-seed AP gain has video-bootstrap 95% intervals of 0.011–0.168 and 0.002–0.152.
The point-estimate effect of consistency on sim→real mIoU changes sign with real exposure; with two training seeds,
this interaction remains descriptive. The probe-gap point estimate grows even as both absolute readout scores improve,
reinforcing why gap size alone is not a quality objective.

#include "navigation.typ"

All visual encoders nearly saturate the initial gap task; this establishes under-hour feasibility, not an
advantage for paired consistency. An actual-city extension shares 1.80M triangles across cameras and uses the
same geometry for GPU collision. Its paired-student policy succeeds on 84/128 held-out episodes (65.6%) after
3.00M transitions in 50.05 minutes, versus 0/128 for each blind heuristic. It does not meet the 80% target.
Matched learned proprioception-only control scores 0/128, while privileged waypoint state reaches 127/128.
This separate pilot uses goal-facing, altitude-stabilized kinematic control, not rotor flight; details are in the supplement.

= Conclusion
In the original fixed-city benchmark, paired-view targets improve every measured novel-appearance readout.
Extra training layouts within the shared assets add little at the same budget and do not improve real transfer
in these two seeds; this does not establish that richer geometry is irrelevant.
INT8 preserves the original student's probe scores closely, but Jetson latency is unmeasured.
Simulation-only consistency gives no clear real-image gain; adding unlabelled real RGB improves readout quality
and car AP without increasing AUROC. GPU control trains within an hour, but actual city navigation remains
challenging; these results do not establish an encoder-ranking advantage. The objectives must be evaluated independently.

*Limitations.* Three training cities and two held-out layouts share one stylised asset library, with a fixed
compute budget. City navigation uses a goal-facing camera and stabilized kinematic velocity, not full flight.
Distillation uses two training seeds, PPO one. Real-image uncertainty rests on only seven videos; embedded latency is unmeasured.


#v(6pt)
#text(size: 9pt)[*References.* LingBot-Vision (Robbyant, 2026); DINOv3 (Meta, 2025); Hinton et al.,
Distilling the Knowledge in a Neural Network (2015); Tobin et al., Domain Randomization (2017); Xiao et al.,
SmoothQuant (2023); Lyu et al., UAVid: A Semantic Segmentation Dataset for UAV Imagery (2020, CC BY-NC-SA 4.0);
Kenney and Poly Haven asset libraries (CC0); #link("https://github.com/NVIDIA/cuda-rust")[NVIDIA CUDA Rust]
(commit `348895e`); #link("https://github.com/gfx-rs/wgpu")[wgpu 30].]
