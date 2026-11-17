# GPU-resident quadrotor visual-goal learning

Task `city-drone-goal-v1`, ABI v4. A quadrotor in the imported city must reach a goal shown only as an image. The native CUDA environment owns dynamics, sensors, rendering and reward; Python owns the learner, evaluation and reporting.

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
  --encoder mean --control velocity --num-envs 64 --eval-envs 384 --width 320 --height 240 \
  --no-gps-fraction 0.25 --aux-coefficient 0.5 \
  --total-steps 3000000 --validation-every 1000000 --wall-seconds 3600 \
  --library .sim-target/training/libsim_env.so --out runs/drone-mean

nix develop path:sim --command .venv/bin/python sim/python/train.py eval \
  --checkpoint runs/drone-mean/best.pt --split test \
  --library .sim-target/training/libsim_env.so --out runs/drone-mean-test

nix develop path:sim --command .venv/bin/python sim/python/train.py probe \
  --encoders mean,baseline,pretrained --train-frames 8192 --test-frames 4096 --epochs 20 \
  --library .sim-target/training/libsim_env.so --out runs/drone-probe

nix develop path:sim --command .venv/bin/python sim/python/train.py memory-check \
  --encoder mean --num-envs 64 --eval-envs 384 --repeats 3 \
  --library .sim-target/training/libsim_env.so --out runs/drone-memory

nix develop path:sim --command .venv/bin/python sim/python/train.py verify-trace \
  --trace runs/drone-profile/trace.json
```

`--control rates` selects the body-rate interface (thrust, roll, pitch, yaw); `velocity` is the default.

Unit tests need no simulator: `cd sim/python && python -m unittest`.

## Task contract

- Control: velocity setpoints by default (PX4 offboard style). Actions are `[forward, right, up, yaw_rate]`, scaled by the native `action_scales` (velocity: 4.0, 4.0, 2.0 m/s and 1.5 rad/s). The native environment clamps the horizontal command to the unit disc. `--control rates` uses `[thrust, roll_rate, pitch_rate, yaw_rate]` with native scales `[MAX_THRUST/2, 3.5, 3.5, 2.0]`. The learner reads scales from native metadata and hardcodes none. The learner stores the unclipped sample for log-probability.
- Hover action: zeros in velocity mode; `(-0.2, 0, 0, 0)` in rates mode (a fixed mild thrust offset; hover thrust is set by native).
- Yaw sign: `yaw_rate` positive is nose left (native convention). The GPS heuristic turns toward the target with this sign.
- Starts are 50% landed and 50% flying. Diagnostics index 19 records the start-landed flag.
- GPS: proprioception lanes 14–16 hold the GPS hint (right, up, −forward) in metres / 32; lane 18 is `gps_available`. With GPS denied the hint is zero. `--no-gps-fraction` sets the probability that an episode denies GPS; evaluation reports GPS available and GPS denied separately.
- Curriculum buckets are `visible`, `partial` and `hidden`. Start weights are (0.8, 0.15, 0.05) with floors (0.10, 0.10, 0.05). Adaptive promotion and demotion use per-bucket EMA success. Evaluation always uses uniform weights. The curriculum tensor is `[w_visible, w_partial, w_hidden, p_no_gps]`.
- Episodes end on success, collision, out-of-bounds, or the native horizon (`horizon`, 450 steps). Outcomes are exclusive, with priority success > collision > out-of-bounds > timeout.
- Catalog: at least 4096 candidate designs. Splits use styles `TRAIN=0`, `VALIDATION=1`, `TEST=2`, with validation seed 10001 and test seed 20002.

## Policy

Frozen encoder tokens are computed once per control tick and reused by every PPO epoch. Visual arms use `GoalAttention` over patch tokens of shape `[N, 15, 20, D]` (320×240 input) with the goal tokens `concat(CLS, mean patch)` from the goal catalog. The attended output is 396 wide: 256 attended features, 8 expected-xy values, 4 peak logits, and a 128-wide global projection. Proprioception (19) is concatenated after it.

Arms (`--encoder`):

| Arm | Actor input | Notes |
|---|---|---|
| `mean` (default) | visual, `run:runs/mean` | paired-consistency encoder |
| `pretrained` | visual, `lingbot:small` | off-the-shelf encoder |
| `baseline` | visual, `run:runs/baseline` | standard-distilled encoder |
| `state` | privileged (21) + proprioception | privileged ceiling; not visual evidence |
| `hint` | proprioception only | no-vision baseline |

The critic always receives privileged (21) + proprioception (19). The actor is a GRU over the trunk output, with a Gaussian head whose log-std is state-independent (initial -0.5). Memory resets at episode starts. Terminal GAE never bootstraps across a done flag.

Auxiliary losses (visual arms only, `--aux-coefficient`, default 0.5) train the goal-attention encoder on privileged image-space labels that the actor never receives: a visibility binary cross-entropy on privileged lane 14 over all frames, and a smooth-L1 localization loss on the target's NDC centre (lanes 18–19) over visible frames only. Invisible frames contribute nothing to localization, and a batch with no visible frames gives exactly zero.

Native tensors are borrowed through DLPack on the native CUDA stream. Reset, step, and rollout never call `.item()`, `.cpu()`, or boolean-mask indexing; readbacks happen only at end-of-rollout and evaluation boundaries.

## Evaluation

Validation and final tests run in both GPS modes: GPS available (`no_gps=0`) and GPS denied (`no_gps=1`). Each mode uses the first completed episode of each environment and must reach at least 256 completed episodes (`MIN_EPISODES`); `--eval-envs` defaults to 384. Reported per mode:

- success with a Wilson 95% interval, collision, out-of-bounds, timeout;
- `mean_steps` (all episodes) and `mean_success_steps` (conditional on success, null when none succeed);
- per-bucket and start-landed/flying breakdowns.

Blind controls run on the same test seed and style: `hover` and `random` (uniform [-1, 1]), plus `gps` (a pure-pursuit controller on the GPS hint) in velocity mode. The `gps` heuristic is not defined in rates mode. Validation selects `best.pt`: validation GPS success when `no_gps_fraction` is 0, otherwise the mean of both modes. Validation skips the blind controls. The reported test uses `best.pt`.

## Localization probe

`train.py probe` is an offline goal-localization probe. It renders frames once per encoder and trains only a probe head on the frozen goal-attention features and auxiliary heads. It reports visibility AUROC and average precision, and centre error in pixels (mean, median, p90) on visible frames, on separate train and test seeds. Frames are shared across encoders. The protocol needs at least 8192 train and 4096 test frames; smaller runs are labelled `"protocol": "smoke"` in `probe.json` and `probe.md`. Probe memory is roughly 2.8 GB of uint8 frames at the protocol size and 320×240.

## Checkpoints

Format `sim-drone-ppo-v2`. A checkpoint records task, ABI version, control mode, encoder spec and fingerprint, policy mode, control rate, city fingerprint, native library hash (provenance only, not compared, so rebuilds stay valid), seeds, PPO settings, `no_gps_fraction` and `aux_coefficient`. Loading rejects v1, v2 and v3 checkpoints and any ABI other than 4.

## Replay and reports

`scripts/render_navigation.py --checkpoint runs/drone-mean/best.pt [--no-gps-fraction 1]` renders one first-episode rollout (`rollout.gif`, `contact.png`, `path.png`, `path.json`, `rollout.json`). `rollout.json` records control mode, action names, GPS mode and outcome. `scripts/show_navigation.py --runs runs/drone-mean runs/drone-pretrained ...` writes `report/city-drone-goal.typ` and `.json` from completed runs only, with GPS-available and GPS-denied columns. Both reject results with fewer than 256 episodes per mode or a different task or ABI.

Replay export (`sim_rl.replay`) writes format `sim-city-replay-v2`: a manifest (including `control_mode`, `action_names`, `no_gps_fraction` and per-environment `gps_available`) plus per-environment states (`world_x, world_y, world_z, quat_x..w, target_x, target_y, target_z, design, clock, landed`) and transitions (`action_names` followed by `reward, done, success, collision, out_of_bounds`), trimmed to the first done frame. Run it from `sim/python` with `PYTHONPATH=. ../../.venv/bin/python -m sim_rl.replay --checkpoint ... --library ... [--no-gps-fraction 0.5]`.

## Resource notes

- Rollout token memory is about 944 MB at 64 environments, D=384, bf16.
- Replay token memory is about 450 × n × 300 × D × 2 bytes; host RGB adds about 1.66 GB at 16 environments.
- Profiling verifies zero host transfers and host waits inside `sim.collect` and `sim.ppo_update` scopes; `verification.json` records the verdict.
