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
  --library .sim-target/training/libsim_env.so --out runs/gpu-profile

nix develop path:sim --command .venv/bin/python sim/python/train.py benchmark \
  --encoder mean --batch-sizes 16 64 128 --width 320 --height 240 \
  --library .sim-target/training/libsim_env.so --out runs/gpu-benchmark

nix develop path:sim --command .venv/bin/python sim/python/train.py train \
  --encoder mean --num-envs 64 --width 320 --height 240 \
  --total-steps 1000000 --wall-seconds 3300 \
  --library .sim-target/training/libsim_env.so --out runs/gpu-mean

nix develop path:sim --command .venv/bin/python sim/python/train.py eval \
  --checkpoint runs/gpu-mean/best.pt \
  --library .sim-target/training/libsim_env.so --out runs/gpu-mean-eval
```

Arms: `pretrained`, `baseline`, `mean`, `state` (privileged ceiling), `proprio` (no-vision baseline), or a real `run:<directory>` checkpoint. Visual arms use the actual frozen encoder online, spatially pooled tokens and a recurrent actor/critic. PPO reuses GPU feature rollouts; it does not run the encoder again for every update epoch. The native ABI tensors are borrowed, never copied, and all model/policy/native work uses the native CUDA stream.

The task has randomized gaps, 20 Hz velocity steering, six 120 Hz rigid-body substeps, swept wall collisions and device-side autoresets. The observed clock makes the 150-step finite horizon terminal; GAE suppresses bootstrap at all terminals and GRU memory resets at episode boundaries. Visual actors receive goal-relative proprioception, not privileged gap coordinates.

Training style 0, validation style 1 and final-test style 2 have disjoint appearance palettes and seed domains. Model selection uses validation only. Final evaluation uses exactly the first completed episode of each of at least 128 environments; center, goal-centering and random-action heuristics run on the same initial episodes. No learned checkpoint is replaced by a random fallback.

The initial engineering acceptance criterion is held-out visual success ≥0.8 and ≥0.2 absolute above the best blind heuristic on ≥128 episodes, within the declared process wall budget. It is a task-level criterion, not real-flight validation or proof of convergence. `summary.json` records actual steps, selected checkpoint, initialization/training/evaluation wall time and incomplete-budget status. Never infer convergence from a timer stopping.

`mean_steps` includes failed episodes; `mean_success_steps` is conditional on success and is null when none succeed. Validation reuses one environment/stream so Torch allocator pools and cuBLAS workspaces do not accumulate per evaluation. The `memory-check` command measures repeated full evaluations after warmup without optimizer updates.

`profile` captures real collection and PPO updates after initialization. Both `sim.collect` and `sim.ppo_update` must pass, and `verification.json` records both verdicts. Launch correlations catch delayed H2D/D2H activities even after the CPU scope closes; host waits in either scope also fail the gate. Initialization, end-batch metric export and checkpoints are separate explicit boundaries. Vulkan draws are included in end-to-end timing, but CUDA Kineto cannot isolate Vulkan render time; the report does not invent it by subtraction.

Profiler traces and checkpoints live under the requested `runs/` directory. Native library and encoder hashes, source hashes, seeds, resolution, package versions, PPO settings and task protocol accompany every training run.
