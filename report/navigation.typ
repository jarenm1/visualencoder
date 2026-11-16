== GPU-owned closed-loop navigation
A separate Rust workspace tests frozen encoders in a first-person random-gap flight task. Hardware: NVIDIA GeForce RTX 5070 Ti. Vulkan/wgpu-hal rasterization shares device-local buffers with NVIDIA Rust CUDA through external-memory FDs and timeline semaphores. CUDA owns dynamics, swept collisions, reward, reset and camera state; a recurrent PPO actor reads frozen visual tokens and proprioception. Frozen encoder inference is bfloat16; the actor and critic use float32. Visual actors see no privileged gap coordinates; the state-only ceiling does. The policy steers laterally with three actions; forward speed and altitude are stabilized. Physics runs at 120 Hz and control at 20 Hz; the observed finite horizon is 150 steps.

All compared runs use 320×240 RGB and the same episode protocol. Selection uses validation seeds and appearance only; final tests use a separate seed domain and palette. Each policy is tested on the first completed episode of 128 environments; the table reports actual executed transitions and end-to-end process time, including initialization and evaluation but excluding compilation.

#figure(table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, center, center, center, center, center),
  stroke: none, table.hline(),
  [*Actor features*], [*Transitions*], [*Minutes*], [*Success [95% CI]* ↑], [*Collision* ↓], [*Goal (s)*],
  table.hline(stroke: 0.5pt),
  [Off-the-shelf ViT-S], [999,424], [25.53], [0.992 [0.957, 0.999]], [0.008], [6.05],
  [Standard-distilled ViT-S], [999,424], [25.71], [1.000 [0.971, 1.000]], [0.000], [6.05],
  [Paired-consistency ViT-S], [999,424], [26.79], [1.000 [0.971, 1.000]], [0.000], [6.05],
  [Privileged state], [999,424], [3.95], [1.000 [0.971, 1.000]], [0.000], [6.05],
  [Proprioception only], [999,424], [3.99], [0.258 [0.190, 0.340]], [0.742], [6.05],
  table.hline(),
), caption: [Independent closed-loop experiment. Goal time is conditional on success (dash: none); forward speed is fixed. Wall times were observed with concurrent distillation, not a controlled encoder-speed comparison. Intervals are Wilson intervals over test episodes, not training-seed uncertainty. 1 training seed(s) shown; no claim of convergence or real-world flight transfer.])

Blind controls on the same test episodes achieve success center 0.273, goal centering 0.273, random 0.273. A separate profiler capture of actual visual collection and PPO updates finds zero host transfers or CUDA host waits inside those ranges. Initialization, post-rollout logging and checkpoint/image export are explicit boundaries outside that claim. This is an initial control-relevance test on simple geometry, not proof of transfer to the detailed city or a physical drone.
