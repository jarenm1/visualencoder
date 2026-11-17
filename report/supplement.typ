// Build: typst compile --root . report/supplement.typ report/supplement.pdf
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

= Original gap-task rollout
#figure(image("figures/navigation.png", width: 100%), caption: [Frames from the paired-consistency visual policy in a
separate held-out rollout. The episode finished successfully after 121 steps. This single inspection episode is not
the aggregate 128-episode result; the recorded GIF and transition metadata accompany the code.])

#pagebreak()

= City renderer and task engineering
#let city = json("../sim/verification.json").city_navigation
The city extension imports the same #city.render_and_collision_triangles transformed GLB triangles for rendering
and swept-sphere collision. Independent agents share static geometry; 16m render chunks are culled on the GPU
and dispatched through persistent indexed indirect commands. RGB uses the original palette materials rather
than the distillation renderer's phototextures, normal maps and shadows.

#figure(table(
  columns: (auto, auto, auto, auto),
  align: (center, center, center, center),
  stroke: none, table.hline(),
  [*Environments*], [*Native images/s*], [*Encoder images/s*], [*Full PPO transitions/s*],
  table.hline(stroke: 0.5pt),
  ..city.city_benchmark.batches.map(row => (
    [#row.environments],
    [#calc.round(row.native_images_per_second)],
    [#calc.round(row.encoder_images_per_second)],
    [#calc.round(row.full_ppo_transitions_per_second)],
  )).flatten(),
  table.hline(),
), caption: [Actual city, RTX 5070 Ti, 320×240 RGB and the frozen paired-consistency encoder.
Native includes kinematic control, exact collision, Vulkan culling/rasterization, packing and CUDA conversion;
it is not isolated draw timing. Concurrent workstation workloads were not controlled.])

The batch-16 profile checks #city.zero_host_staging.collection.cuda_runtime_calls CUDA runtime calls in collection
and #city.zero_host_staging.ppo_update.cuda_runtime_calls in PPO updates, with zero host transfers or host waits
in either scope. GPU packing, conversion and rollout feature copies remain; initialization, batch-end metrics
and artifact export are outside the claim. Vulkan validation reports no errors.

Before learning, a privileged route oracle completes 128/128 episodes in each district without collision.
World-frame velocity error diffusion prevents discrete-heading pursuit from cutting inside clear path chords.
This is task-solvability evidence, not a visual-policy result. Eight blind-forward test episodes all collide;
hover leaves position unchanged through the finite horizon.

#figure(image("figures/city-cameras.png", width: 100%), caption: [Actual GPU-rendered initial test-route views at
6m altitude. Cameras face their final goals, which lie behind actual geometry. These are unselected initialized
views, not learned-policy trajectories.])

#pagebreak()
#include "city-navigation.typ"

#figure(image("figures/city-navigation.png", width: 100%), caption: [The fixed first test episode of the actual
trained city visual policy, reaching its goal after 234 steps (11.7s), without collision. Frames and the
exported world trajectory show one inspection episode, not the aggregate success estimate. The image encoder
is frozen; only the recurrent PPO actor and critic are trained.])
