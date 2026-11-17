# GPU-resident visual learning

Use the root Python environment (`nix develop` from the repository root installs it) and the simulator's CUDA/Vulkan shell when launching native learning.

```sh
# Build outside the timed training process.
cd sim
nix develop --command cargo oxide build --arch sm_120 \
  --device-codegen-crate sim_physics,sim_env -- -p sim-env --profile training
cd ..

nix develop path:sim --command .venv/bin/python sim/python/train.py profile \
  --encoder mean --num-envs 16 --width 320 --height 240 \
  --library .sim-target/training/libsim_env.so --out runs/city-profile

nix develop path:sim --command .venv/bin/python sim/python/train.py benchmark \
  --encoder mean --batch-sizes 16 32 64 --width 320 --height 240 \
  --library .sim-target/training/libsim_env.so --out runs/city-benchmark

nix develop path:sim --command .venv/bin/python sim/python/train.py train \
  --encoder mean --num-envs 16 --width 320 --height 240 \
  --total-steps 3000000 --validation-every 262144 --wall-seconds 3300 \
  --library .sim-target/training/libsim_env.so --out runs/city-mean

nix develop path:sim --command .venv/bin/python sim/python/train.py eval \
  --checkpoint runs/city-mean/best.pt \
  --library .sim-target/training/libsim_env.so --out runs/city-mean-eval
```

Arms: `pretrained`, `baseline`, `mean`, `state` (privileged ceiling), `proprio` (no-vision baseline), or a real `run:<directory>` checkpoint. Visual arms use the actual frozen encoder online, spatially pooled tokens and a recurrent actor/critic. PPO reuses GPU feature rollouts; it does not run the encoder again for every update epoch. The native ABI tensors are borrowed, never copied, and all model/policy/native work uses the native CUDA stream.

At the policy input, goal displacement is scaled by 32m, vertical displacement by 6m, and velocity by 3m/s; clock and previous-action values retain unit scale. ABI tensors remain in physical units. Checkpoints record these scales.

The task is real-city block navigation: fixed altitude 6m, hover or eight goal-frame velocity commands at 3m/s, six exact swept-sphere collision substeps at 120Hz per 20Hz action, and GPU autoresets. The level camera automatically faces the final goal. This is kinematic navigation, not rotor/aerodynamic flight. The observed clock makes the 320-step horizon terminal; GAE suppresses bootstrap at every terminal and GRU memory resets at episode boundaries.

Training, validation and test occupy separate 64m city districts, each with 128 independently seeded routes. Routes are 12–38m, require a blocked direct line, and have geometry-checked waypoint paths with 0.85m planning clearance for the 0.35m collision sphere. Reward is `0.2 × path-progress − 0.01` per step, with +5 goal/−5 collision. Visual actors receive goal-relative proprioception, never route waypoints; only the explicit state ceiling receives privileged hints.

Model selection uses validation only. Final evaluation uses exactly the first completed episode of each of at least 128 environments; hover, goal-centering and random-action heuristics use the same initial routes. `oracle-diagnostic` separately checks privileged route following and is **not learned or visual evidence**. No learned checkpoint is replaced by a fallback.

All resource-creating commands accept `--city assets/city` (the default). ABI v2 and `sim-city-ppo-v2` checkpoints record scene-content fingerprints, controls, route splits, rewards and native library hashes. Historical gap checkpoints are rejected rather than silently evaluated as city policies.

The initial engineering acceptance criterion is held-out visual success ≥0.8 and ≥0.2 absolute above the best blind heuristic on ≥128 episodes, within the declared process wall budget. It is a task-level criterion, not real-flight validation or proof of convergence. `summary.json` records actual steps, selected checkpoint, initialization/training/evaluation wall time and incomplete-budget status. Never infer convergence from a timer stopping.

`mean_steps` includes failed episodes; `mean_success_steps` is conditional on success and is null when none succeed. Validation reuses one environment/stream so Torch allocator pools and cuBLAS workspaces do not accumulate per evaluation. The `memory-check` command measures repeated full evaluations after warmup without optimizer updates.

`profile` captures real collection and PPO updates after initialization. Both `sim.collect` and `sim.ppo_update` must pass, and `verification.json` records both verdicts. Launch correlations catch delayed H2D/D2H activities even after the CPU scope closes; host waits in either scope also fail the gate. Initialization, end-batch metric export and checkpoints are separate explicit boundaries. Vulkan draws are included in end-to-end timing, but CUDA Kineto cannot isolate Vulkan render time; the report does not invent it by subtraction.

Profiler traces and checkpoints live under the requested `runs/` directory. Native library and encoder hashes, source hashes, seeds, resolution, package versions, PPO settings and task protocol accompany every training run.

## Measured city pipeline

RTX 5070 Ti, 320×240 RGB, frozen paired-consistency encoder; concurrent workstation workloads were not controlled. Native timing includes kinematic control, exact collision, Vulkan culling/rasterization, sensor packing and CUDA conversion—not an isolated draw time.

| Batch | Native images/s | Encoder images/s | Full PPO transitions/s |
|---:|---:|---:|---:|
| 16 | 8,755 | 1,626 | 1,201 |
| 32 | 9,616 | 1,539 | 1,201 |
| 64 | 11,735 | 1,367 | 1,080 |

Measurements: `runs/city-benchmark-final/benchmark.json`. The final batch-16 trace checked 41,300 collection and 4,062 PPO CUDA runtime calls: zero host transfers or host waits in either scope (`runs/city-profile-final/verification.json`). GPU packing/conversion and rollout feature copies still occur.

The privileged error-diffused route oracle completed 128/128 episodes in each split with zero collisions (`runs/city-oracle-quantized/oracle-diagnostic.json`). This checks task solvability, not learned-policy performance. Simple nearest-heading pursuit cut inside clear route chords; world-frame velocity error diffusion removes that quantization error without changing geometry or routes.

Actual camera views: `runs/city-smoke/city-cameras.png`. Eight blind-forward episodes collided; hover preserved position exactly over the finite horizon. The same run reported no Vulkan validation errors.

Each city learner executed 2,999,808 transitions with matched PPO settings and one training seed:

| Policy inputs | Successes / 128 | Collisions / 128 | Process minutes |
|---|---:|---:|---:|
| Frozen paired encoder + proprioception | 84 | 24 | 50.05 |
| Proprioception only | 0 | 0 | 13.34 |
| Privileged waypoint state + proprioception | 127 | 1 | 13.57 |

Runs: `runs/city-{mean,proprio,state}-320-seed0`. Times include initialization and evaluation. Validation selected the visual checkpoint at 1,572,864 steps. All three blind heuristics scored 0/128. The visual 80% success target was **not met**; these results do not claim convergence or an advantage over other visual encoders. Privileged waypoints are not visual-policy inputs.

The fixed first test rollout succeeds in 234 steps (11.7s): `reports/experiments/city-navigation-rollout/rollout.gif`, with actual world-path diagnostics beside it. Tables and intervals: `report/city-navigation.typ` and `.json`; architecture/timing/rollout evidence is compiled into `report/supplement.pdf`.
