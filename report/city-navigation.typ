== GPU-owned city block navigation
A separate Rust workspace tests frozen encoders on the actual imported city geometry at stabilized altitude. Hardware: NVIDIA GeForce RTX 5070 Ti. Vulkan/wgpu-hal rasterization shares device-local buffers with NVIDIA Rust CUDA through external-memory FDs and timeline semaphores. CUDA owns kinematic agent state, swept collisions, reward, reset and camera state; a recurrent PPO actor reads frozen visual tokens and proprioception. Frozen encoder inference is bfloat16; the actor and critic use float32. Visual actors receive only images and final-goal proprioception, never route waypoints or collision geometry; the explicit state ceiling sees next-waypoint state. The 9-action controller commands hover or eight unit-normalized horizontal velocity headings in the goal-facing camera frame, with automatic goal yaw. Kinematic/collision substeps run at 120 Hz and control at 20 Hz; the finite terminal horizon is 320 steps. Collision uses exact swept-sphere tests against the same transformed city triangles used for rendering. City RGB uses palette base colours, not the legacy phototexture/normal-map/shadow pipeline. Dense reward measures progress along collision-free route waypoints, with time cost and terminal success/collision terms. Scene content fingerprint: #raw("sha256:62e67df07b2bff3c13aed8983e8dd3dbc5393d33c4f98c3895b8f73cb43fb217").

All compared runs use 320×240 RGB and the same episode protocol. Selection uses a validation route district; train, validation and final-test routes occupy spatially disjoint city districts with fixed independent catalogs. Each policy is tested on the first completed episode of at least 128 environments; the table reports actual executed transitions and end-to-end process time, including initialization and evaluation but excluding compilation.

#figure(table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, center, center, center, center, center),
  stroke: none, table.hline(),
  [*Actor features*], [*Transitions*], [*Minutes*], [*Success [95% CI]* ↑], [*Collision* ↓], [*Goal (s)*],
  table.hline(stroke: 0.5pt),
  [Paired-consistency ViT-S], [2,999,808], [50.05], [0.656 [0.570, 0.733]], [0.188], [10.14],
  [Proprioception only], [2,999,808], [13.34], [0.000 [0.000, 0.029]], [0.000], [—],
  [Privileged waypoint state], [2,999,808], [13.57], [0.992 [0.957, 0.999]], [0.008], [10.49],
  table.hline(),
), caption: [Independent city navigation experiment. Goal time is conditional on success (dash: none); commanded horizontal speed and altitude are stabilized. Process times include the observed concurrent workload, not a controlled encoder-speed comparison. Intervals are Wilson intervals over test episodes, not training-seed uncertainty. 1 training seed(s) shown; no claim of convergence or real-world flight transfer.])

Blind controls on the same test episodes achieve success hover 0.000, goal centering 0.000, random 0.000. The profiling command separately verifies zero host transfers or CUDA host waits inside actual visual collection and PPO update ranges. Initialization, post-rollout logging and checkpoint/image export are explicit boundaries outside that claim. This is a rendered city navigation experiment with goal-facing stabilized control, not full rotor flight or proof of transfer to a physical drone.
