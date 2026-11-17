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
  --encoder mean --num-envs 128 --eval-envs 256 \
  --total-steps 6000000 --validation-every 1000000 --wall-seconds 3300 \
  --out runs/drone4-mean-seed0

nix develop path:sim --command .venv/bin/python sim/python/train.py eval \
  --checkpoint runs/drone-mean/best.pt --split test \
  --library .sim-target/training/libsim_env.so --out runs/drone-mean-test

nix develop path:sim --command .venv/bin/python sim/python/train.py probe --out runs/drone-probe

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
- Catalog: 65,536 episodes per split (each bucket ≥512). Targets are 1.5–2.5 m designs, starts 4–20 m away; success radius is the target's enclosing sphere + 1 m. Splits use styles `TRAIN=0`, `VALIDATION=1`, `TEST=2`, with validation seed 10001 and test seed 20002.

## Policy

Frozen encoder tokens are computed once per control tick and reused by every PPO epoch. Visual arms use `TargetSaliency` over patch tokens `[N, 300, D]` (320×240 input, 16 px patches): each patch scores a goal-free linear term plus a low-rank (rank 64) token × goal bilinear term, where the goal is `concat(CLS, mean patch)` of the design's reference image. A "not in view" logit from CLS and goal completes a softmax over 301 classes. The actor receives the saliency-expected target NDC (2), the peak patch logit (1), P(in view) (1), saliency-pooled tokens (64) and a projected CLS + 2×3 pooled context (128), then proprioception (19).

Arms (`--encoder`):

| Arm | Actor input | Notes |
|---|---|---|
| `mean` (default) | visual, `run:runs/mean` | paired-consistency encoder |
| `pretrained` | visual, `lingbot:small` | off-the-shelf encoder |
| `baseline` | visual, `run:runs/baseline` | standard-distilled encoder |
| `state` | privileged (21) + proprioception | privileged ceiling; not visual evidence |
| `hint` | proprioception only | no-vision baseline |

The critic always receives privileged (21) + proprioception (19). The actor is a GRU over the trunk output, with a Gaussian head whose log-std is state-independent (initial -0.5). Memory resets at episode starts. Terminal GAE never bootstraps across a done flag.

The auxiliary loss (visual arms only, `--aux-coefficient`, default 0.5) is cross-entropy over those 301 classes: the patch containing the target centre when privileged lane 14 says it is visible, otherwise "not in view". The actor never receives these privileged labels. The learning curve logs `aux_patch_accuracy` (exact patch on visible frames).

Native tensors are borrowed through DLPack on the native CUDA stream. Reset, step, and rollout never call `.item()`, `.cpu()`, or boolean-mask indexing; readbacks happen only at end-of-rollout and evaluation boundaries.

## Evaluation

Validation and final tests run in both GPS modes: GPS available (`no_gps=0`) and GPS denied (`no_gps=1`). Each mode uses the first completed episode of each environment and must reach at least 256 completed episodes (`MIN_EPISODES`); `--eval-envs` defaults to 384. Reported per mode:

- success with a Wilson 95% interval, collision, out-of-bounds, timeout;
- `mean_steps` (all episodes) and `mean_success_steps` (conditional on success, null when none succeed);
- per-bucket and start-landed/flying breakdowns.

Blind controls run on the same test seed and style: `hover` and `random` (uniform [-1, 1]), plus `gps` (a pure-pursuit controller on the GPS hint) in velocity mode. The `gps` heuristic is not defined in rates mode. Validation selects `best.pt`: validation GPS success when `no_gps_fraction` is 0, otherwise the mean of both modes. Validation skips the blind controls. The reported test uses `best.pt`.

## Localization probe

`train.py probe` is the offline goal-localization gate run before RL. Each frame is the start view of an independent catalog episode (8,192 train-split, 4,096 test-split frames; test districts and designs are held out), shared across encoders. A fresh `TargetSaliency` is trained on each frozen encoder with the policy's saliency loss (identical seed, 20 epochs, lr 1e-3). It reports visibility AUROC/AP, exact-patch and within-one-patch localization, and pixel error of the predicted patch. Frames from correlated random-action rollouts let probes memorize (≈90% train vs 25% test patch accuracy) and are not used; an earlier learned-position query attention head also memorized and was replaced.

Measured (`runs/drone-probe-saliency`, test split):

| Encoder | Visibility AUROC | Exact patch | Within 1 patch | Median error |
|---|---|---|---|---|
| `pretrained` (LingBot ViT-S) | 0.862 | 0.539 | 0.821 | 8.4 px |
| `baseline` (standard distilled) | 0.894 | 0.629 | 0.861 | 7.6 px |
| `mean` (paired consistency) | 0.897 | 0.620 | 0.856 | 7.7 px |

## Measured navigation (velocity control, one seed, held-out test split)

256 episodes per GPS mode, uniform visibility buckets; Wilson 95% intervals. Wall times include initialization, training and evaluation on an RTX 5070 Ti shared with an unrelated GPU workload (not controlled). The camera arm stopped at its wall budget with fewer steps than the controls.

| Actor | Steps | Minutes | GPS success | Visible / partial / hidden | No-GPS success | Crash (GPS) |
|---|---|---|---|---|---|---|
| Camera + `mean` encoder | 1.98M | 49.5 | 0.449 [0.389, 0.510] | 0.55 / 0.61 / 0.22 | 0.430 [0.371, 0.491] | 0.25 |
| GPS hint + sensors only (learned) | 6.00M | 34.5 | 0.199 [0.155, 0.252] | 0.23 / 0.33 / 0.07 | 0.078 [0.051, 0.118] | 0.48 |
| Privileged target state (ceiling) | 6.00M | 34.7 | 0.805 [0.752, 0.849] | 0.91 / 0.87 / 0.64 | 0.746 [0.689, 0.796] | 0.18 |
| Scripted GPS pursuit | — | — | 0.043 | 0.02 / 0.09 / 0.03 | 0.000 | 0.42 |

The camera policy was trained only with GPS available; its no-GPS score is zero-shot. Results are single-seed and budget-limited, not convergence claims; the camera arm does not yet include the `pretrained`/`baseline` encoder comparison.

## Checkpoints

Format `sim-drone-ppo-v2`. A checkpoint records task, ABI version, control mode, encoder spec and fingerprint, policy mode, control rate, city fingerprint, native library hash (provenance only, not compared, so rebuilds stay valid), seeds, PPO settings, `no_gps_fraction` and `aux_coefficient`. Loading rejects v1, v2 and v3 checkpoints and any ABI other than 4.

## Replay and reports

`scripts/render_navigation.py --checkpoint runs/drone-mean/best.pt [--no-gps-fraction 1]` renders one first-episode rollout (`rollout.gif`, `contact.png`, `path.png`, `path.json`, `rollout.json`). `rollout.json` records control mode, action names, GPS mode and outcome. `scripts/show_navigation.py --runs runs/drone-mean runs/drone-pretrained ...` writes `report/city-drone-goal.typ` and `.json` from completed runs only, with GPS-available and GPS-denied columns. Both reject results with fewer than 256 episodes per mode or a different task or ABI.

Replay export (`sim_rl.replay`) writes format `sim-city-replay-v2`: a manifest (including `control_mode`, `action_names`, `no_gps_fraction` and per-environment `gps_available`) plus per-environment states (`world_x, world_y, world_z, quat_x..w, target_x, target_y, target_z, design, clock, landed`) and transitions (`action_names` followed by `reward, done, success, collision, out_of_bounds`), trimmed to the first done frame. Run it from `sim/python` with `PYTHONPATH=. ../../.venv/bin/python -m sim_rl.replay --checkpoint ... --library ... [--no-gps-fraction 0.5]`.

## Resource notes

- Rollout token memory is about 944 MB at 64 environments, D=384, bf16.
- Replay token memory is about 450 × n × 300 × D × 2 bytes; host RGB adds about 1.66 GB at 16 environments.
- Profiling verifies zero host transfers and host waits inside `sim.collect` and `sim.ppo_update` scopes; `verification.json` records the verdict.
