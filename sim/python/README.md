# GPU-resident quadrotor visual-goal learning

Task `city-drone-goal-v2`, ABI v5. A quadrotor in the imported city must fly to an object shown only as a goal image — a crop of some camera view of that object — and hold a standoff in front of it. Targets are the city's own buildings, vehicles, trees and props. The native CUDA environment owns dynamics, sensors, rendering, appearance randomization and reward; Python owns the learner, evaluation and reporting.

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

- Control: velocity setpoints by default (PX4 offboard style). Actions are `[forward, right, up, yaw_rate]`, scaled by the native `action_scales` (velocity: 4.0, 4.0, 2.0 m/s and 1.5 rad/s). The native environment clamps the horizontal command to the unit disc. `--control rates` uses `[thrust, roll_rate, pitch_rate, yaw_rate]` with native scales `[MAX_THRUST/2, 3.5, 3.5, 2.0]`. The learner reads scales from native metadata and hardcodes none. The policy is a tanh-squashed Gaussian: the simulator receives `tanh(raw)` and rollouts store the pre-squash sample, whose log-probability includes the tanh Jacobian. (The v4 policy clipped an unbounded mean; once the mean left [-1, 1] its gradient vanished and failures sat at full yaw rate.)
- Hover action: zeros in velocity mode; `(-0.2, 0, 0, 0)` in rates mode (a fixed mild thrust offset; hover thrust is set by native).
- Yaw sign: `yaw_rate` positive is nose left (native convention). The GPS heuristic turns toward the target with this sign.
- Starts are 50% landed and 50% flying. Diagnostics index 19 records the start-landed flag.
- GPS: proprioception lanes 14–16 hold the GPS hint to the target box centre (right, up, −forward) in metres / 32, with a per-episode 3–15 m error; lane 18 is `gps_available`. With GPS denied the hint is zero. `--no-gps-fraction` sets the probability that an episode denies GPS; evaluation reports GPS available and GPS denied separately.
- Curriculum buckets are `visible`, `partial` and `hidden`. Start weights are (0.8, 0.15, 0.05) with floors (0.10, 0.10, 0.05). Adaptive promotion and demotion use per-bucket EMA success. Evaluation always uses uniform weights. The curriculum tensor is `[w_visible, w_partial, w_hidden, p_no_gps, appearance]`.
- Targets: placed objects of `assets/city/layout.json` of class building, vehicle, vegetation or prop whose box lies inside the split's district; vegetation and props must stand ≥2 m (box gap) from any object of their class so a crop identifies one object. Episodes draw the class uniformly, then the object.
- Success (standoff): within 6 m of the target's bounding box, slower than 1 m/s and facing it — the box point nearest the camera ray (the anchor) inside the frustum and unoccluded — for 15 consecutive steps (0.5 s). Reward: progress in box distance (no reward below 3 m), −0.01 per step, action smoothness, body-rate and yaw-rate costs (−0.02 per step at full yaw rate), a standoff bonus while within 6 m and facing of 0.1 × (1 − speed/3 m/s) per step, +10 success, −10 crash, −5 out-of-bounds. (With an all-or-nothing hold bonus the policy found targets but circled them at ~2.6 m/s and never braked.)
- Episodes end on success, collision, out-of-bounds, or the native horizon (`horizon`, 450 steps). Outcomes are exclusive, with priority success > collision > out-of-bounds > timeout.
- Catalog: 65,536 episodes per split (each bucket ≥512); starts 8–30 m from the target box. Each episode fixes a goal reference camera: the (levelled) start view in half of the visible-start episodes — the operator taps the object in the first frame — otherwise a level camera 5–25 m from the box at 1.5–12 m height (drone-like; bird's-eye references made unrecognizable goals) that sees at least half of it. The goal crop is the box's projection grown 15% per side and squared; the object must span ≥8% of the frame and the crop ≤80%, otherwise the view is rejected, so every goal shows one recognizable object. Splits use styles `TRAIN=0`, `VALIDATION=1`, `TEST=2` (disjoint districts), with validation seed 10001 and test seed 20002.
- Appearance: a post-render CUDA kernel randomizes, per catalog entry, material hue/saturation/value, sky gradient, depth fog, exposure, white balance, contrast, gamma and sensor noise (`--appearance` strength, default 1; 0 is the plain render). An episode and its goal view share one appearance. Sun direction and shadows are fixed.

## Policy

Frozen encoder tokens are computed once per control tick and reused by every PPO epoch. Visual arms use `TargetSaliency` over patch tokens `[N, 300, D]` (320×240 input, 16 px patches): each patch scores a goal-free linear term plus a low-rank (rank 64) token × goal bilinear term, where the goal is `concat(CLS, mean patch)` of the episode's goal crop. Goal embeddings are built once per environment: the native `render_goals` renders every catalog entry's reference view, the learner crops it (`features.crop_goals`, the same crop any operator or detector box would use) and encodes it into a `[65536, 2D]` bf16 table (~50 s at 128 envs); each step gathers the row of `env.catalog_index`. A "not in view" logit from CLS and goal completes a softmax over 301 classes. The actor receives the saliency-expected target NDC (2), the peak patch logit (1), P(in view) (1), saliency-pooled tokens (64) and a projected CLS + 2×3 pooled context (128), then proprioception (19).

Arms (`--encoder`):

| Arm | Actor input | Notes |
|---|---|---|
| `mean` (default) | visual, `run:runs/mean` | paired-consistency encoder |
| `pretrained` | visual, `lingbot:small` | off-the-shelf encoder |
| `baseline` | visual, `run:runs/baseline` | standard-distilled encoder |
| `state` | privileged (29) + proprioception | privileged ceiling; not visual evidence |
| `hint` | proprioception only | no-vision baseline |

Encoder inference (`--encoder-precision`, recorded in the checkpoint's `preprocessing`; `eval` and replay read it back):

| Precision | Path | Encoder µs/img (batch 128, RTX 5070 Ti) | Patch cosine to FP32 (mean / min) |
|---|---|---|---|
| `fp16` (default) | TensorRT 11 FP16 engine, strongly typed ONNX export (RoPE stays FP32) | 190 | 1.00000 / 0.99966 |
| `bf16` | `torch.compile` under bf16 autocast; numerics of pre-TensorRT checkpoints | 278 | 0.99997 / 0.99948 |
| (previous eager bf16) | removed | 602 | 0.99997 / 0.99912 |

The engine is built once per encoder fingerprint, input size, GPU and TensorRT version (about 20 s) and cached in `$XDG_CACHE_HOME/visenc/tensorrt`; batches above 256 run in chunks. `torch.compile` needs the `sim` shell's `TRITON_*` paths.

`train.py benchmark --encoder mean` (64 rollout steps, collect + update):

| Path | Transitions/s at 64 envs | Transitions/s at 128 envs | Encoder ms/step at 128 |
|---|---|---|---|
| eager bf16 (previous) | 1,413 | 1,334 | 76.8 |
| `bf16` (`torch.compile`) | 2,164 | 2,391 | 34.8 |
| `fp16` (TensorRT) | 2,949 | 2,987 | 24.2 |

The critic always receives privileged (29) + proprioception (19). The actor is a GRU over the trunk output, with a tanh-squashed Gaussian head whose log-std is state-independent (initial -0.5). Memory resets at episode starts. Terminal GAE never bootstraps across a done flag. The entropy bonus is a reparameterized estimate of the squashed distribution's entropy; the pre-squash Gaussian entropy grows without bound and drove the first v5 run into saturated bang-bang actions.

The auxiliary loss (visual arms only, `--aux-coefficient`, default 0.5) is cross-entropy over those 301 classes against a soft target: uniform over the patches whose centres fall inside the target box's projection (privileged lanes 25–28, always including the anchor patch) when privileged lane 14 says the target is visible, otherwise "not in view". A single anchor patch was an arbitrary label for large objects (offline within-one-patch localization 0.38). The actor never receives these privileged labels. The learning curve logs `aux_patch_accuracy`: the fraction of visible frames whose peak patch lies on the target's projection.

Native tensors are borrowed through DLPack on the native CUDA stream. Reset, step, and rollout never call `.item()`, `.cpu()`, or boolean-mask indexing; readbacks happen only at end-of-rollout and evaluation boundaries.

## Evaluation

Validation and final tests run in both GPS modes: GPS available (`no_gps=0`) and GPS denied (`no_gps=1`). Each mode uses the first completed episode of each environment and must reach at least 256 completed episodes (`MIN_EPISODES`); `--eval-envs` defaults to 384. Reported per mode:

- success with a Wilson 95% interval, collision, out-of-bounds, timeout;
- `mean_steps` (all episodes) and `mean_success_steps` (conditional on success, null when none succeed);
- per-bucket and start-landed/flying breakdowns.

Blind controls run on the same test seed and style: `hover` and `random` (uniform [-1, 1]), plus `gps` (a pure-pursuit controller on the GPS hint) in velocity mode. The `gps` heuristic is not defined in rates mode. Validation selects `best.pt`: validation GPS success when `no_gps_fraction` is 0, otherwise the mean of both modes. Validation skips the blind controls. The reported test uses `best.pt`.

## Localization probe

`train.py probe` is the offline goal-localization gate run before RL. Each frame is the start view of an independent catalog episode (8,192 train-split, 4,096 test-split frames; test districts are held out), shared across encoders. Each frame's goal is its catalog entry's goal-crop embedding. A fresh `TargetSaliency` is trained on each frozen encoder with the policy's saliency loss (identical seed, 20 epochs, lr 1e-3). It reports visibility AUROC/AP, exact-patch and within-one-patch localization and pixel error of the predicted patch, overall and per target class; `--appearance` sets the randomization strength. Frames from correlated random-action rollouts let probes memorize (≈90% train vs 25% test patch accuracy) and are not used; an earlier learned-position query attention head also memorized and was replaced.

Measured on ABI v4 (designed 1.5–2.5 m shapes, held-out designs; `runs/drone-probe-saliency`, test split):

| Encoder | Visibility AUROC | Exact patch | Within 1 patch | Median error |
|---|---|---|---|---|
| `pretrained` (LingBot ViT-S) | 0.862 | 0.539 | 0.821 | 8.4 px |
| `baseline` (standard distilled) | 0.894 | 0.629 | 0.861 | 7.6 px |
| `mean` (paired consistency) | 0.897 | 0.620 | 0.856 | 7.7 px |

## ABI v5 results (city-object targets)

Camera + `mean` encoder (bf16), velocity control, seed 0, 128 envs, 2.54M steps in the 3300 s budget (49.5 min incl. startup, validation and test; `runs/drone5-mean-seed0`). Held-out test district, 256 episodes per GPS mode, uniform visibility buckets, appearance randomization on; Wilson 95% intervals.

| Actor | GPS success | Visible / partial / hidden | No-GPS success | Crash (GPS) | Out of bounds | Timeout |
|---|---|---|---|---|---|---|
| Camera + `mean` encoder | 0.070 [0.045, 0.108] | 0.12 / 0.06 / 0.02 | 0.078 [0.051, 0.118] | 0.46 | 0.12 | 0.35 |
| Scripted GPS pursuit | 0.070 | — | 0.000 | 0.64 | — | — |
| Hover / random | 0.000 / 0.000 | — | 0.000 / 0.000 | 0.00 / 0.16 | — | — |

By class (GPS): buildings 0.086, vegetation 0.096, vehicles 0.059, props 0.033. Validation success rose 0.031 → 0.055 → 0.086 at 1.0M / 2.0M / 2.54M steps, so the run stopped at its budget, not a plateau. Training-episode success reached 0.36, but the adaptive curriculum never left 85% visible starts (promotion needs 0.7 visible success), while evaluation draws the buckets uniformly. With GPS the camera policy only ties the scripted pursuit; without GPS it keeps 0.078 where the pursuit scores 0, so its target finding is visual. Crashes are the dominant failure. No spinning remains: a 1M-step replay had no frames at full yaw rate (v4 failures: 82–98%).

Earlier v5 attempts, kept for the record: the Gaussian entropy bonus drove saturated bang-bang actions (entropy 3.7 → 4.1, success → 0); single-patch anchor labels and bird's-eye, full-frame goal crops kept localization near chance (offline within-one-patch 0.38; box-region labels and drone-like crops reach peak-on-target 0.58 offline, 0.85–0.9 in training); and an all-or-nothing hold bonus left the policy circling found targets at ~2.6 m/s (`runs/drone5-mean-seed0-binaryhold`, partial).

Live play: `nix develop --command nix develop path:sim --command .venv/bin/python sim/python/live.py --checkpoint runs/drone5-mean-seed0/best.pt [--gemma]` serves http://127.0.0.1:8790/: the policy flies one held-out episode in real time; drag a box around any object (or type an instruction with `--gemma`) to make it the goal and, when it matches a city object, the scored target, with a fresh 15 s clock. Boxes of objects that project partly behind the camera cannot be matched, so very close objects fall back to whatever lies behind them.

## ABI v4 results (designed targets; superseded task)

Velocity control, one seed, held-out test split. 256 episodes per GPS mode, uniform visibility buckets; Wilson 95% intervals. Wall times include initialization, training and evaluation on an RTX 5070 Ti shared with an unrelated GPU workload (not controlled). The camera arm stopped at its wall budget with fewer steps than the controls.

| Actor | Steps | Minutes | GPS success | Visible / partial / hidden | No-GPS success | Crash (GPS) |
|---|---|---|---|---|---|---|
| Camera + `mean` encoder | 1.98M | 49.5 | 0.449 [0.389, 0.510] | 0.55 / 0.61 / 0.22 | 0.430 [0.371, 0.491] | 0.25 |
| GPS hint + sensors only (learned) | 6.00M | 34.5 | 0.199 [0.155, 0.252] | 0.23 / 0.33 / 0.07 | 0.078 [0.051, 0.118] | 0.48 |
| Privileged target state (ceiling) | 6.00M | 34.7 | 0.805 [0.752, 0.849] | 0.91 / 0.87 / 0.64 | 0.746 [0.689, 0.796] | 0.18 |
| Scripted GPS pursuit | — | — | 0.043 | 0.02 / 0.09 / 0.03 | 0.000 | 0.42 |

The camera policy was trained only with GPS available; its no-GPS score is zero-shot. Results are single-seed and budget-limited, not convergence claims; the camera arm does not yet include the `pretrained`/`baseline` encoder comparison.

### Language goals via Gemma 4 on ABI v4 (no retraining)

Gemma 4 E4B-it chose the goal: it saw the 8 held-out reference images as numbered options plus one instruction and answered an option number (bf16 on GPU, 5.6 GB per-layer embedding table kept on CPU; ~0.75 s per answer). The chosen design's embedding replaced the true goal; the camera policy above flew 256 test episodes per condition. (Code: commit `b32da5c`, `sim/python/gemma_goal.py`; ABI v5 replaced design picking with grounding, below.)

| Goal source | Pick accuracy (8) | Success |
|---|---|---|
| True goal | 1.00 | 0.449 [0.389, 0.510] |
| Gemma, colour only ("Go to the green one") | 0.75 | 0.418 [0.359, 0.479] |
| Gemma, riddle ("…shaped like a donut") | 0.75 | 0.367 [0.311, 0.428] |
| Gemma, full name ("Fly to the checkered green octahedron") | 0.625 | 0.344 [0.288, 0.404] |
| Wrong design (ablation) | 0 | 0.121 |

Every Gemma miss confused two designs that share colour and pattern but differ in shape (octahedron/torus, cube/cylinder). Colour-only instructions are ambiguous in this split (two green, two blue designs) but scored higher than full names.

## Checkpoints

Format `sim-drone-ppo-v3`. A checkpoint records task, ABI version, control mode, encoder spec and fingerprint, policy mode, control rate, catalog size, city fingerprint, native library hash (provenance only, not compared, so rebuilds stay valid), seeds, PPO settings, `no_gps_fraction`, `aux_coefficient` and `appearance`. Loading rejects earlier formats and any ABI other than 5.

## Replay and reports

`scripts/render_navigation.py --checkpoint runs/drone-mean/best.pt [--no-gps-fraction 1]` renders one first-episode rollout (`rollout.gif`, `contact.png`, `path.png`, `path.json`, `rollout.json`). `rollout.json` records control mode, action names, GPS mode and outcome. `scripts/show_navigation.py --runs runs/drone-mean runs/drone-pretrained ...` writes `report/city-drone-goal.typ` and `.json` from completed runs only, with GPS-available and GPS-denied columns. Both reject results with fewer than 256 episodes per mode or a different task or ABI.

Replay export (`sim_rl.replay`, entry `sim/python/replay.py --checkpoint ... --out ... --num-envs 32 --split test`) writes format `sim-city-replay-v3`: a manifest (including `control_mode`, `action_names`, `no_gps_fraction`, `appearance`, and per environment `gps_available`, `target_class`, layout `object` and the goal crop `env-XXX-goal.png`) plus per-environment states (pose, target box centre and half extents, catalog index, target class and object, clock, landed, and the anchor labels) and transitions (`action_names` followed by `reward, done, success, collision, out_of_bounds`), trimmed to the first done frame. `--target-class NAME` keeps episodes whose target has that layout class. The site shows the goal crop beside the controls.

Language goals by grounding (`sim/python/gemma_goal.py`): `frames` renders the first-person start frames of a fresh held-out batch; `ground --instruction TEXT` asks Gemma 4 to box the instructed object in each frame and matches the box to the target-class city object whose projected bounding box overlaps it most (IoU ≥ 0.3; occlusion is not modelled). `replay.py --goal-boxes DIR/boxes.json` then flies those episodes: the goal is the crop of the start frame at Gemma's box (the crop an operator tap would give) and the native `retarget` call makes success score the matched object; unmatched boxes still steer the policy but the episode keeps its catalog target (`grounding.scored = false`).

Typed goals in the replay site: start the server inside the root dev shell (Gemma needs its libraries) with `nix develop --command .venv/bin/python sim/replay/serve.py --data <recording> --live [--checkpoint runs/drone5-mean-seed0/best.pt]`. Each instruction runs frames → ground → record on `--live-envs` (default 32) held-out environments with a fresh random seed and switches the page to the recording; episodes where Gemma boxed nothing are dropped. Each recording is served under its own `/replay/<recording>/` URLs, and the page follows the server's current recording.

## Resource notes

- Rollout token memory is about 944 MB at 64 environments, D=384, bf16.
- Replay token memory is about 450 × n × 300 × D × 2 bytes; host RGB adds about 1.66 GB at 16 environments.
- Profiling verifies zero host transfers and host waits inside `sim.collect` and `sim.ppo_update` scopes; `verification.json` records the verdict.
