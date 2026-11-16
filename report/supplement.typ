#set document(title: "Supplement: Appearance-Robust Vision Encoders for Drones", author: "Jaren McHugh")
#set page(paper: "us-letter", margin: 0.8in, numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show table: set text(size: 9pt)
#show figure.caption: set text(size: 9pt)

#align(center)[#text(size: 15pt, weight: "bold")[Supplementary Results] \
Jaren McHugh]

= Additional ablations
#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    [*Variant*], [*Pose R²*], [*Depth err.*], [*Seg. mIoU*], [*Retrieval*],
    table.hline(stroke: 0.5pt),
    [Baseline], [0.465], [0.242], [0.788], [0.747],
    [No appearance randomization], [0.506], [0.246], [0.769], [0.769],
    [DINOv3-L as teacher], [−0.134], [0.208], [0.838], [0.800],
    [Random-init student], [−0.239], [0.387], [0.534], [0.439],
    [Paired-view consistency], [*0.574*], [*0.201*], [0.821], [*0.951*],
    table.hline(),
  ),
  caption: [Original novel-appearance benchmark; one element changes from standard distillation in each ablation. Descriptive results, not significance tests.],
)

Canonical-only training and standard randomized distillation perform similarly on these novel-appearance probes;
randomization alone does not reproduce the paired-target gains. A DINOv3 teacher improves segmentation while
student pose R² becomes negative. Random initialization is much worse at this training budget, not evidence that
learning from scratch is impossible at a larger budget.

= Real-image qualitative predictions
#figure(image("figures/real.png", width: 100%), caption: [Real UAVid frames: ground truth (black = classes with no
simulator counterpart) and predictions of segmentation probes fit only on simulator renders. These examples are not the aggregate accuracy estimate.])

= Precision export
The original simulation-only paired-target student is exported to ONNX for TensorRT FP16. NVIDIA ModelOpt
inserts SmoothQuant on linear layers, calibrated on 512 renders. LayerNorm inputs remain FP16: residual-stream
outliers up to ±400 make the default configuration destructive (feature cosine 0.58).

INT8 feature cosine to FP32 is 0.998 and every measured probe changes by at most 0.009
(largest: novel retrieval 0.957 → 0.948); FP16 leaves reported probe scores unchanged.
This check does not cover the newly adapted/layout-trained students. No Jetson latency is claimed.
Tools include PyTorch, Transformers, ModelOpt, ONNX Runtime, TensorRT, Blender and raylib, with Nix/uv environments.

= Geometry and appearance support
#figure(image("figures/geometry.png", width: 100%), caption: [Columns: original city, two additional training layouts,
and two held-out layouts. Top: canonical appearance; bottom: novel appearance at the same pose within each layout.
Representative urban views are selected by ground-truth coverage and camera height, not encoder predictions.])

#include "geometry-extra.typ"

= Actual trained-policy rollout
#figure(image("figures/navigation.png", width: 100%), caption: [Frames from the paired-consistency visual policy in a
separate held-out rollout. The episode finished successfully after 121 steps. This single inspection episode is not
the aggregate 128-episode result; the recorded GIF and transition metadata accompany the code.])
