# GPU-resident quadrotor visual-goal learning

Task `city-drone-goal-v1`, ABI v3. A quadrotor in the imported city must reach a goal shown only as an image. The native CUDA environment owns dynamics, sensors, rendering and reward; Python owns the learner, evaluation and reporting.

Use the root Python environment (`nix develop` from the repository root installs it) and the simulator's CUDA/Vulkan shell when launching native learning.

```sh
# Build outside the timed training process.
cd sim
nix develop --command cargo oxide build --arch sm_120 \
  --device-codegen-crate sim_physics,sim_env -- -p sim-env -p sim-smoke --profile training
cd ..

nix develop path:sim --command .venv/bin/python sim/python/train.py profile \
  --encoder mean --num-envs 64 --width 320 --height 240 \
  --library .sim-target/training/libsim_env.so --out runs/drone-profile

nix develop path:sim --command .venv/bin/python sim/python/train.py benchmark \
  --encoder mean --batch-sizes 64 128 --width 320 --height 240 \
  --library .sim-target/training/libsim_env.so --out runs/drone-benchmark

nix develop path:sim --command .venv/bin/python sim/python/train.py train \
  --encoder mean --num-envs 64 --eval-envs 384 --width 320 --height 240 \
  --total-steps 3000000 --validation-every 1000000 --wall-seconds 3600 \
  --library .sim-target/training/libsim_env.so --out runs/drone-mean

nix develop path:sim --command .venv/bin/python sim/python/train.py eval \
  --checkpoint runs/drone-mean/best.pt --split test \
  --library .sim-target/training/libsim_env.so --out runs/drone-mean-test

nix develop path:sim --command .venv/bin/python sim/python/train.py memory-check \
  --encoder mean --num-envs 64 --eval-envs 384 --repeats 3 \
  --library .sim-target/training/libsim_env.so --out runs/drone-memory

nix develop path:sim --command .venv/bin/python sim/python/train.py verify-trace \
  --trace runs/drone-profile/trace.json
```

Unit tests need no simulator: `cd sim/python && python -m unittest`.

## Task contract

- Dynamics: quadrotor flight at the native control rate (`control_hz` in the checkpoint metadata). Actions are `[throttle, roll, pitch, yaw]`, each clamped to [-1, 1] before the environment applies it. The learner stores the unclipped sample for log-probability.
- Starts are 50% landed and 50% flying. Diagnostics index 19 records the start-landed flag.
- Curriculum buckets are `visible`, `partial` and `hidden`. Start weights are (0.8, 0.15, 0.05) with floors (0.10, 0.10, 0.05). Adaptive promotion and demotion use per-bucket EMA success. Evaluation always uses uniform weights.
- Episodes end on success, collision, out-of-bounds, or the native horizon (`horizon`, 450 steps). Outcomes are exclusive, with priority success > collision > out-of-bounds > timeout.
- Catalog: at least 4096 candidate designs. Splits use styles `TRAIN=0`, `VALIDATION=1`, `TEST=2`, with validation seed 10001 and test seed 20002.

## Policy

Frozen encoder tokens are computed once per control tick and reused by every PPO epoch. Visual arms use `GoalAttention` over patch tokens of shape `[N, 15, 20, D]` (320×240 input) with the goal tokens `concat(CLS, mean patch)` from the goal catalog. The attended output is 396 wide: 256 attended features, 8 expected-xy values, 4 peak logits, and a 128-wide global projection. Proprioception (18) is concatenated after it.

Arms (`--encoder`):

| Arm | Actor input | Notes |
|---|---|---|
| `mean` (default) | visual, `run:runs/mean` | paired-consistency encoder |
| `pretrained` | visual, `lingbot:small` | off-the-shelf encoder |
| `baseline` | visual, `run:runs/baseline` | standard-distilled encoder |
| `state` | privileged (18) + proprioception | privileged ceiling; not visual evidence |
| `hint` | proprioception only | no-vision baseline |

The critic always receives privileged + proprioception. The actor is a GRU over the trunk output, with a Gaussian head whose log-std is state-independent (initial -0.5). Memory resets at episode starts. Terminal GAE never bootstraps across a done flag.

Native tensors are borrowed through DLPack on the native CUDA stream. Reset, step, and rollout never call `.item()`, `.cpu()`, or boolean-mask indexing; readbacks happen only at end-of-rollout and evaluation boundaries.

## Evaluation

Final tests use the first completed episode of each of at least 384 environments (`--eval-envs`, enforced). Reported:

- success with a Wilson 95% interval, collision, out-of-bounds, timeout;
- `mean_steps` (all episodes) and `mean_success_steps` (conditional on success, null when none succeed);
- per-bucket and start-landed/flying breakdowns.

Blind controls `hover` (fixed hover action) and `random` (uniform [-1, 1]) run on the same test seed and style. Validation selects `best.pt` on validation success; the reported test uses `best.pt`.

## Checkpoints

Format `sim-drone-ppo-v1`. A checkpoint records task, ABI version, encoder spec and fingerprint, policy mode, control rate, city fingerprint, native library hash (provenance only, not compared, so rebuilds stay valid), seeds and PPO settings. Loading rejects v1/v2 checkpoints and ABI v2 libraries.

## Replay and reports

`scripts/render_navigation.py --checkpoint runs/drone-mean/best.pt` renders one first-episode rollout (`rollout.gif`, `contact.png`, `path.png`, `path.json`, `rollout.json`). `scripts/show_navigation.py --runs runs/drone-mean runs/drone-pretrained ...` writes `report/city-drone-goal.typ` and `.json` from completed runs only. Both reject results with fewer than 384 episodes or a different task or ABI.

Replay export (`sim_rl.replay`) writes format `sim-city-replay-v2`: a manifest plus per-environment states (`world_x, world_y, world_z, quat_x..w, target_x, target_y, target_z, design, clock, landed`) and transitions (`throttle, roll, pitch, yaw, reward, done, success, collision, out_of_bounds`), trimmed to the first done frame. Run it from `sim/python` with `PYTHONPATH=. ../../.venv/bin/python -m sim_rl.replay --checkpoint ... --library ...`.

## Resource notes

- Rollout token memory is about 944 MB at 64 environments, D=384, bf16.
- Replay token memory is about 450 × n × 300 × D × 2 bytes; host RGB adds about 1.66 GB at 16 environments.
- Profiling verifies zero host transfers and host waits inside `sim.collect` and `sim.ppo_update` scopes; `verification.json` records the verdict.
